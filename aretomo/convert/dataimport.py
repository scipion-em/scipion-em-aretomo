# **************************************************************************
# *
# * Authors:     Scipion Team (scipion@cnb.csic.es) [1]
# *
# * [1] National Center of Biotechnology, CSIC, Spain
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

import os.path
import numpy as np
import logging

logger = logging.getLogger(__name__)

import pyworkflow.utils as pwutils
from pwem.objects import CTFModel

with pwutils.weakImport('tomo'):
    from tomo.objects import CTFTomo


class AretomoCtfParser:
    """ Load and parse CTF estimated with Aretomo. """

    def __init__(self, protocol):
        self.protocol = protocol

    def parseTSDefocusFile(self, ts, fileName, output):
        """ Parse tilt-series ctf estimation file.
        :param ts: input tilt-series
        :param fileName: input file to be parsed
        :param output: output CTFTomoSeries
        """
        psdFile = pwutils.replaceExt(fileName, 'mrc')
        psdFile = psdFile if os.path.exists(psdFile) else None
        ctfResult = self.readAretomoCtfOutput(fileName)
        counter = 0

        for i, ti in enumerate(ts):
            ctf = CTFModel()
            if ti.isEnabled():
                self._getCtfTi(ctf, ctfResult, item=counter, psdFile=psdFile)
                counter += 1
            else:
                ctf.setWrongDefocus()

            newCtfTomo = CTFTomo.ctfModelToCtfTomo(ctf)
            # The enabled attribute has to be managed here because the method ctfModelToCtfTomo can't copy that
            # attribute as it is a python boolean instead of a Scipion Boolean object. This is because that attribute
            # belongs to the class Object, which is above in the hierarchy than the Scipion types, so it can't follow
            # their setters and getters logic, failing when trying to copy the mentioned attribute
            if not ti.isEnabled():
                newCtfTomo.setEnabled(False)
            newCtfTomo.setIndex(i + 1)
            newCtfTomo.setAcquisitionOrder(ti.getAcquisitionOrder())
            output.append(newCtfTomo)

    @staticmethod
    def _getCtfTi(ctfModel, ctfArray, item=0, psdFile=None):
        """ Set values for the ctfModel from an input list.
        :param ctfModel: output CTF model
        :param ctfArray: array with CTF values
        :param item: which row to use from ctfArray
        :param psdFile: psd file name, including the path.
        """
        values = ctfArray[item]
        ctfPhaseShift = 0
        ctfFit = -999
        resolution = -999
        if np.isnan(values).any(axis=0) or values[1] < 0 or values[2] < 0:
            logger.debug(f"Invalid CTF values: {values}")
            ctfModel.setWrongDefocus()
        else:
            # 1 - micrograph number
            # 2 - defocus 1 [A]
            # 3 - defocus 2
            # 4 - azimuth of astigmatism
            # 5 - additional phase shift [radian]
            # 6 - cross correlation
            # 7 - spacing (in Angstroms) up to which CTF rings were fit successfully.
            # From a conversation with B. K. regarding the resolution value: "I think it is column #7. I believe that
            # Shawn was hesitant to call it resolution, as in his view the resolution might be higher, but it states
            # the resolution to which the ctf fit is reliable. In the end, I believe that the indication by ctffind
            # is similar"
            tiNum, defocusV, defocusU, defocusAngle, ctfPhaseShift, ctfFit, resolution = values
            ctfModel.setStandardDefocus(defocusU, defocusV, defocusAngle)
        ctfModel.setFitQuality(ctfFit)
        ctfModel.setResolution(resolution)
        if ctfPhaseShift != 0:
            ctfModel.setPhaseShift(np.rad2deg(ctfPhaseShift))
        if psdFile:
            ctfModel.setPsdFile(f"{item + 1}@" + psdFile)

    @staticmethod
    def readAretomoCtfOutput(filename):
        """ Reads an Aretomo CTF file and loads it as a numpy array with the following information for each line:
        Columns:
        #1 - micrograph number
        #2 - defocus 1 [A]
        #3 - defocus 2
        #4 - azimuth of astigmatism
        #5 - additional phase shift [radian]
        #6 - cross correlation
        #7 - spacing (in Angstroms) up to which CTF rings were fit successfully.
        :param filename: input file to read.
        :return: a numpy array with the CTF values.
        """
        if os.path.exists(filename):
            return np.loadtxt(filename, dtype=float, comments='#')
        else:
            logger.error(f"Warning: Missing file: {filename}")
            return None
