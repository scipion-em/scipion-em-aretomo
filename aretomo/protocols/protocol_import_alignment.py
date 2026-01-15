# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk) [1]
# *              Federico P. de Isidro Gomez (fp.deisidro@cnb.csic.es) [2]
# *              Alberto Garcia Mena (alberto.garcia@cnb.csic.es) [2]
# *              Scipion Team (scipion@cnb.csic.es) [2]
# *
# * [1] MRC Laboratory of Molecular Biology (MRC-LMB)
# * [2] Centro Nacional de Biotecnologia, CSIC, Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 3 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import logging
import os
import traceback
from collections import Counter

import numpy as np
import time
from typing import List, Tuple, Union, Optional

from pwem import ALIGN_2D
from pyworkflow.protocol import params, STEPS_PARALLEL
from pyworkflow.constants import PROD
from pyworkflow.object import Set, String, Pointer
from pyworkflow.protocol import ProtStreamingBase
import pyworkflow.utils as pwutils
from pwem.protocols import EMProtocol
from pwem.objects import Transform, CTFModel
from pwem.emlib.image import ImageHandler
from pyworkflow.utils import Message, cyanStr, getExt, createLink, redStr
from tomo.protocols import ProtTomoBase
from tomo.objects import (Tomogram, TiltSeries, TiltImage,
                          SetOfTomograms, SetOfTiltSeries, SetOfCTFTomoSeries, CTFTomoSeries, CTFTomo)

from .. import Plugin
from ..convert.convert import getTransformationMatrix, readAlnFile, writeAlnFile
from ..convert.dataimport import AretomoCtfParser
from ..constants import RECON_SART, LOCAL_MOTION_COORDS, LOCAL_MOTION_PATCHES

logger = logging.getLogger(__name__)

OUT_TS = "TiltSeries"
OUT_TOMO = "Tomograms"
OUT_CTFS = "CTFTomoSeries"
FAILED_TS = 'FailedTiltSeries'
EVEN = '_even'
ODD = '_odd'
MRC_EXT = '.mrc'
MRCS_EXT = '.mrcs'


class ProtAreTomoImportAlignment(EMProtocol):
    """ Import AreTomo alignment from an existing run"""
    _label = 'import alignment'
    _devStatus = PROD
    _possibleOutputs = {OUT_TS: SetOfTiltSeries}
    stepsExecutionMode = STEPS_PARALLEL

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('inputSetOfTiltSeries',
                      params.PointerParam,
                      pointerClass=SetOfTiltSeries,
                      important=True,
                      label='Input set of Tilt-Series')
        form.addParam('alnPathFormat',
                      params.StringParam,
                      important=True,
                      label='Alignment file pattern')



    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._insertFunctionStep(self.createOutputStep)

    # --------------------------- STEPS functions -----------------------------
    def createOutputStep(self):
        inputSetOfTiltSeries = self._getInputTiltSeries()
        outputSetOfTiltSeries = SetOfTiltSeries.create(self.getPath())
        outputSetOfTiltSeries.copyInfo(inputSetOfTiltSeries)

        for tiltSeries in inputSetOfTiltSeries.iterItems():
            tsId = tiltSeries.getTsId()
            alnFilename = self._getAlignmentFile(tsId)

            try:
                aln = readAlnFile(alnFilename)
            except OSError:
                self.warning(f'No alignment file was found for {tsId}')
                continue
            
            alignmentMatrix = getTransformationMatrix(aln.imod_matrix)
            sectionToIndex = dict(zip(aln.sections, range(len(aln.sections))))

            newTiltSeries = TiltSeries()
            newTiltSeries.copyInfo(tiltSeries)
            newTiltSeries.setSamplingRate(tiltSeries.getSamplingRate())
            newTiltSeries.setAlignment2D()
            outputSetOfTiltSeries.append(newTiltSeries)
            
            for tiltImage in tiltSeries.iterItems():
                newTiltImage = tiltImage.clone()
                newTiltImage.copyInfo(tiltImage, copyId=True, copyTM=False)
                transform = Transform()
                section = tiltImage.getIndex()
                index = sectionToIndex.get(section, None)

                if index is not None:
                    acq = tiltImage.getAcquisition()
                    acq.setTiltAxisAngle(aln.tilt_axes[index])
                    newTiltImage.setAcquisition(acq)
                    newTiltImage.setTiltAngle(aln.tilt_angles[index])

                    m = alignmentMatrix[:, :, index]
                    transform.setMatrix(m)
                else:
                    newTiltImage.setEnabled(False)
                    transform.setMatrix(np.identity(3))

                newTiltImage.setTransform(transform)
                newTiltImage.setSamplingRate(tiltImage.getSamplingRate())
                newTiltSeries.append(newTiltImage)
                
                
                acq = newTiltSeries.getAcquisition()
                acq.setTiltAxisAngle(aln.tilt_axes[0])
                newTiltSeries.setAcquisition(acq)

                outputSetOfTiltSeries.update(newTiltSeries)
                outputSetOfTiltSeries.write()

        self._defineOutputs(**{OUT_TS: outputSetOfTiltSeries})
        self._defineSourceRelation(self.inputSetOfTiltSeries, outputSetOfTiltSeries)

    # --------------------------- INFO functions ------------------------------
    def _warnings(self):
        warnings = []
        
        inputSetOfTiltSeries = self._getInputTiltSeries()
        for tiltSeries in inputSetOfTiltSeries.iterItems():
            tsId = tiltSeries.getTsId()
            alnFilename = self._getAlignmentFile(tsId)
            if not os.exists(alnFilename):
                warnings.append(f'No alignment file for {alnFilename}')
    
        return warnings
    
    # --------------------------- UTILS functions -----------------------------
    def _getInputTiltSeries(self) -> SetOfTiltSeries:
        return self.inputSetOfTiltSeries.get()
    
    def _getAlignmentFile(self, tsId: str) -> str:
        formatter = self.alnPathFormat.get()
        return formatter.format(TS=tsId)
    