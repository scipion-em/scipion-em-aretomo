# **************************************************************************
# *
# * Authors:     Scipion Team (scipion@cnb.csic.es)
# *
# * Centro Nacional de Biotecnologia, CSIC, Spain
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
import typing
from os.path import join, exists
import numpy as np
import time
from typing import List, Tuple, Union, Optional
from pwem import ALIGN_2D
from pyworkflow.protocol import STEPS_PARALLEL, BooleanParam, IntParam, PointerParam, LEVEL_ADVANCED, \
    EnumParam, FloatParam, StringParam, GPU_LIST
from pyworkflow.constants import BETA
from pyworkflow.object import Set, String, Pointer
from pyworkflow.protocol import ProtStreamingBase
import pyworkflow.utils as pwutils
from pwem.protocols import EMProtocol
from pwem.objects import Transform, CTFModel
from pwem.emlib.image import ImageHandler
from pyworkflow.utils import cyanStr, getExt, createLink, redStr
from tomo.protocols import ProtTomoBase
from tomo.objects import (Tomogram, TiltSeries, TiltImage,
                          SetOfTomograms, SetOfTiltSeries, SetOfCTFTomoSeries, CTFTomoSeries, CTFTomo)

from .. import Plugin
from ..convert.convert import getTransformationMatrix, readAlnFile, writeAlnFile
from ..convert.dataimport import AretomoCtfParser
from ..constants import RECON_SART

logger = logging.getLogger(__name__)

OUT_TS = "TiltSeries"
OUT_TOMO = "Tomograms"
OUT_CTFS = "CTFTomoSeries"
FAILED_TS = 'FailedTiltSeries'
EVEN = '_even'
ODD = '_odd'
MRC_EXT = '.mrc'

# Form parameters
DO_ALI_TS = 'doAliTs'
DO_TOMO_REC = 'makeTomo'
DO_CTF_EST = 'doEstimateCtf'
ALIGN_Z = 'alignZ'
TOMO_THK = 'tomoThickness'
EXTRA_Z_HEIGHT = 'extraZHeight'
RECON_RANGE = 'reconRange'

# Tilt-axis refinement options
NO_REFINE = 0
REFINE_3_DEG = 1
REFINE_10_DEG = 2


class ProtAreTomo3(EMProtocol, ProtTomoBase, ProtStreamingBase):
    """ Protocol for fiducial-free alignment and reconstruction for tomography available in streaming. """
    _label = 'tilt-series align, ctf estimation and tomogram reconstruction'
    _devStatus = BETA
    _possibleOutputs = {OUT_TS: SetOfTiltSeries,
                        OUT_TOMO: SetOfTomograms,
                        OUT_CTFS: SetOfCTFTomoSeries}
    stepsExecutionMode = STEPS_PARALLEL

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)
        self.TS_read = []
        self.badTsAliMsg = String()
        self.badTomoRecMsg = String()
        self.excludedViewsMsg = String()
        self.failedItems = []

        # --------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        form.addSection(label='Tilt-series')
        form.addParam('inputSetOfTiltSeries',
                      PointerParam,
                      pointerClass='SetOfTiltSeries',
                      important=True,
                      label='Input set of Tilt-Series',
                      help='If you choose to skip alignment, the input '
                           'tilt-series are expected to be already aligned.')

        form.addParam(DO_ALI_TS, BooleanParam,
                      default=False,
                      label='Align the tilt-series?',
                      help='You can skip alignment if you just want to '
                           'reconstruct a tomogram from already '
                           'aligned tilt-series.')

        form.addParam(ALIGN_Z, IntParam, default=800,
                      condition=DO_ALI_TS,
                      important=True,
                      label='Volume height for alignment (voxels)',
                      help='Specifies Z height (*unbinned*) of the temporary volume '
                           'reconstructed for projection matching as part '
                           'of the alignment process. This value plays '
                           'an important role in alignment accuracy. This '
                           'Z height should be always smaller than tomogram '
                           'thickness and should be close to the sample '
                           'thickness.\n'
                           '*IMPORTANT*: if set to 0, AreTomo3 will estimate the sample '
                           'thickness and use that value during the alignment.')

        form.addParam('refineTiltAngles',
                      EnumParam,
                      condition=DO_ALI_TS,
                      choices=['No', 'Measure only', 'Measure and correct'],
                      display=EnumParam.DISPLAY_COMBO,
                      label="Refine tilt angles?",
                      default=1,
                      help="You have three options:\na) Disable measure and correction\n"
                           "b) Measure only (default). Correction is done during alignment but not "
                           "for final reconstruction\nc) Measure and correct\n\n"
                           "Occasionally, the measurement is erroneous and can impair the "
                           "alignment accuracy. Please note that the orientation of the missing "
                           "wedge will be changed as a result of the correction of tilt offset. "
                           "For subtomogram averaging, tomograms reconstructed from tilt "
                           "series collected within the same tilt range may have different "
                           "orientations of missing wedges.")

        form.addParam('refineTiltAxis', EnumParam,
                      condition=DO_ALI_TS,
                      choices=['No',
                               'Refine within +/- 10 deg.',
                               'Refine within +/- 3 deg.'],
                      display=EnumParam.DISPLAY_COMBO,
                      label="Refine tilt axis angle?",
                      default=NO_REFINE)

        form.addParam('outImod', EnumParam,
                      display=EnumParam.DISPLAY_COMBO,
                      condition=DO_ALI_TS,
                      expertLevel=LEVEL_ADVANCED,
                      choices=['No', 'Relion 4', 'Warp', 'Save locally aligned TS'],
                      default=0,
                      label="Generate extra IMOD output?",
                      help="0 - No\n"
                           "1 - generate IMOD files for Relion 4\n"
                           "2 - generate IMOD files for Warp\n"
                           "3 - generate IMod files when the aligned tilt series "
                           "is used as the input for Relion 4 or WARP")

        group = form.addGroup('Patch-based local alignment')
        form.addParam('doLocalAli', BooleanParam,
                      condition=DO_ALI_TS,
                      default=False,
                      label='Do local alignment?')
        line = group.addLine("Patches", condition=f'{DO_ALI_TS} and doLocalAli')
        line.addParam('patchX', IntParam,
                      default=4,
                      label='X')
        line.addParam('patchY', IntParam,
                      default=4,
                      label='Y')

        form.addParam('darkTol', FloatParam,
                      default=0.7,
                      condition=DO_ALI_TS,
                      important=True,
                      label="Dark tolerance",
                      help="Set tolerance for removing dark images. The range is "
                           "in (0, 1). The default value is 0.7. "
                           "The higher value is more restrictive.")

        form.addSection(label='Tomograms')
        form.addParam(DO_TOMO_REC, BooleanParam,
                      default=True,
                      label='Reconstruct the tomograms?',
                      help='You can skip tomogram reconstruction, so that input '
                           'tilt-series will be only aligned.')

        form.addParam('doEvenOdd', BooleanParam,
                      label='Reconstruct the odd/even tomograms?',
                      default=False,
                      condition=DO_TOMO_REC)

        form.addParam('binFactor', IntParam,
                      default=2,
                      condition=DO_TOMO_REC,
                      label='Binning',
                      important=True,
                      help='Binning for the output volume.')

        form.addParam(TOMO_THK, IntParam,
                      condition=DO_TOMO_REC,
                      important=True,
                      default=1200,
                      label='Tomogram thickness unbinned (voxels)',
                      help='Z height of the reconstructed volume in '
                           '*unbinned* voxels.')

        form.addParam(EXTRA_Z_HEIGHT, IntParam,
                      condition=f'{DO_TOMO_REC} and {ALIGN_Z} <= 0 and {TOMO_THK} <= 0',
                      default=300,
                      label='Extra volume z height for reconstruction (voxels)',
                      help='AreTomo3 will use the estimated sample thickness plus an extra space '
                           'above and below the sample to reconstruct the tomograms. '
                           'The value for the extra space is given by the current parameter,.')

        form.addParam('reconMethod', EnumParam,
                      choices=['SART', 'WBP'],
                      default=RECON_SART,
                      condition=DO_TOMO_REC,
                      display=EnumParam.DISPLAY_HLIST,
                      label="Reconstruction method",
                      help="Choose either SART or weighted back "
                           "projection (WBP).")

        line = form.addLine("SART options",
                            condition=f'{DO_TOMO_REC} and reconMethod=={RECON_SART}')
        line.addParam('SARTiter', IntParam,
                      default=15,
                      label='iterations')
        line.addParam('SARTproj', IntParam,
                      default=5,
                      label='projections per subset')

        line = form.addLine("Tilt-angle range for reconstruction",
                            condition=DO_TOMO_REC,
                            help='It specifies the min and max tilt angles from which a 3D '
                                 'volume will be reconstructed. Any tilt image whose tilt-ange '
                                 'is outside this range is excluded in the reconstruction.')
        line.addParam('minTiltRec', IntParam,
                      default=-60,
                      label='Min tilt-angle (deg.)')
        line.addParam('maxTiltRec', IntParam,
                      default=60,
                      label='Max tilt-angle (deg.)')

        form.addParam('flipInt', BooleanParam,
                      default=False,
                      condition=DO_TOMO_REC,
                      label="Flip intensity?",
                      help="By default, the reconstructed volume "
                           "and the input tilt series use the same grayscale "
                           "that makes dense structures dark.")

        form.addParam('flipVol', BooleanParam,
                      condition=DO_TOMO_REC,
                      default=True,
                      label="Flip volume?",
                      help="Set to Yes when making a tomogram and No when "
                           "making a tilt-series. This way the output orientation "
                           "will be similar to IMOD.")

        form.addSection(label='CTF')
        form.addParam(DO_CTF_EST, BooleanParam,
                       default=True,
                       label='Estimate the CTF?',
                       condition=DO_ALI_TS)

        form.addParam('doCorrCtf', BooleanParam,
                       condition=f'{DO_ALI_TS} and {DO_CTF_EST}',
                       label='Do local CTF correction?',
                       default=False,
                       help='If set to Yes, local CTF correction is performed on '
                            'the raw tilt series. It enables local CTF deconvolution '
                            'of each tilt image. A tilt image is first divided into tiles. '
                            'Each tile has its own CTF based on its location from the tilt-axis. '
                            'CTF deconvolution is done on each tile. Then CTF deconvolved tiles '
                            'are put together to form the CTF deconvolved image.')

        form.addParam('doPhaseShiftSearch', BooleanParam,
                       default=False,
                       label='Do phase shift estimation?',
                       condition=DO_CTF_EST)

        linePhaseShift = form.addLine('Phase shift range (deg.)',
                                       condition='doPhaseShiftSearch',
                                       help="Search range of the phase shift (start, end).")
        linePhaseShift.addParam('minPhaseShift', IntParam,
                                default=0,
                                label='min',
                                condition='doPhaseShiftSearch')
        linePhaseShift.addParam('maxPhaseShift', IntParam,
                                default=0,
                                label='max',
                                condition='doPhaseShiftSearch')

        form.addSection(label='Extra options')
        # form.addParam('doDW', BooleanParam,
        #               default=False,
        #               label="Do dose-weighting?")
        # form.addParam('dosePerRawFrame', FloatParam,
        #               default=0.,
        #               condition='doDW',
        #               label='Dose per raw frame (e/Å^2)',
        #               help='It is the value of the electron dose per each raw frame, '
        #                    'not the accumulated dose on sample. Thus, the users need to '
        #                    'know the number of frames in a movie.')
        form.addParam('extraParams', StringParam,
                      default='',
                      label='Additional parameters',
                      help="Extra command line parameters. See AreTomo help.")

        form.addHidden(GPU_LIST, StringParam,
                       default='0',
                       label="Choose GPU IDs",
                       help="")

        form.addParallelSection(threads=2, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def stepsGeneratorStep(self) -> None:
        """
        This step should be implemented by any streaming protocol.
        It should check its input and when ready conditions are met
        call the self._insertFunctionStep method.
        """
        closeSetStepDeps = []
        inTsSet = self._getSetOfTiltSeries()
        self.readingOutput()
        outputsToCheck = self._getOutputsToCheck()

        if self._getAttribVal(ALIGN_Z) <= 0:
            logger.info(cyanStr('Volume height for alignment is <= 0 -> the sample thickness will be estimated.'))
            if self._getAttribVal(TOMO_THK) <= 0:
                logger.info(cyanStr(f'Tomogram thickness is <= 0 -> the estimated sample thickness plus an extra '
                                    f'space above and below of {self._getAttribVal(EXTRA_Z_HEIGHT)} voxels '
                                    f'to reconstruct the tomogram.'))

        while True:
            listTSInput = inTsSet.getTSIds()
            if not inTsSet.isStreamOpen() and self.TS_read == listTSInput:
                logger.info(cyanStr('Input set closed, all items processed\n'))
                self._insertFunctionStep(self.closeOutputSetStep,
                                         outputsToCheck,
                                         prerequisites=closeSetStepDeps,
                                         needsGPU=False)
                break
            for ts in inTsSet.iterItems():
                if ts.getTsId() not in self.TS_read and ts.getSize() > 0:  # Avoid processing empty TS (before the Tis are added)
                    tsId = ts.getTsId()
                    with self._lock:
                        fName = ts.getFirstItem().getFileName()
                    args = (tsId, fName)
                    convertInput = self._insertFunctionStep(self.convertInputStep, *args,
                                                            prerequisites=[],
                                                            needsGPU=False)
                    runAreTomo = self._insertFunctionStep(self.runAreTomoStep, *args,
                                                          prerequisites=[convertInput],
                                                          needsGPU=True)
                    createOutputS = self._insertFunctionStep(self.createOutputStep, *args,
                                                             prerequisites=[runAreTomo],
                                                             needsGPU=False)
                    closeSetStepDeps.append(createOutputS)
                    logger.info(cyanStr(f"Steps created for TS_ID: {tsId}"))
                    self.TS_read.append(tsId)

            time.sleep(10)
            if inTsSet.isStreamOpen():
                with self._lock:
                    inTsSet.loadAllProperties()  # refresh status for the streaming

    # --------------------------- STEPS functions -----------------------------
    def convertInputStep(self, tsId: str, tsFn: str):
        try:
            logger.info(cyanStr(f'tsId = {tsId} -> converting the inputs...'))
            ts = self.getTsFromTsId(tsId)
            doAlignTs = self._getAttribVal(DO_ALI_TS)

            extraPrefix = self._getExtraPath(tsId)
            tmpPrefix = self._getTmpPath(tsId)
            pwutils.makePath(*[tmpPrefix, extraPrefix])
            outputTsFileName = self.getFilePath(tsFn, tmpPrefix, tsId, ext=MRC_EXT)

            if not doAlignTs:
                if self._getAttribVal(DO_TOMO_REC):
                    createLink(tsFn, outputTsFileName)
                    alnFile = self.getAlnFile(tsFn, tsId)
                    writeAlnFile(ts, tsFn, alnFile)
            else:
                if self.doEvenOdd.get():
                    outputTsFnEven = self.getFilePathEven(tsFn, tmpPrefix, tsId, ext=MRC_EXT)
                    outputTsFnOdd = self.getFilePathOdd(tsFn, tmpPrefix, tsId, ext=MRC_EXT)
                    ts.applyTransformToAll(outputTsFileName,
                                           outFileNamesEvenOdd=[outputTsFnEven, outputTsFnOdd])
                else:
                    ts.applyTransform(outputTsFileName)

                # Generate angle file:
                # AreTomo3 assumes tilt angles are saved in a text file that shares the same file name but ended with
                # either .rawtilt of _TLT.txt. The tilt series file and the associated tilt angle file must be in the
                # same directory.
                presentAcqOrders = ts.getTsPresentAcqOrders()
                angleFilePath = self.getFilePath(tsFn, tmpPrefix, tsId, ext=".rawtlt")
                self._genAretomo3TltFile(ts,
                                         angleFilePath,
                                         presentAcqOrders=presentAcqOrders)

                # if self.alignZfile.hasValue():
                #     alignZfile = self.alignZfile.get()
                #     if exists(alignZfile):
                #         self.perTsAlignZ = self.readThicknessFile(alignZfile)

        except Exception as e:
            self.failedItems.append(tsId)
            logger.error(redStr(f'tsId = {tsId} -> input conversion failed with the exception -> {e}'))
            logger.error(traceback.format_exc())

    def runAreTomoStep(self, tsId: str, tsFn: str):
        """ Call AreTomo with the appropriate parameters. """
        if tsId in self.failedItems:
            return
        logger.info(cyanStr(f'tsId ={tsId} -> running AreTomo...'))
        try:
            ts = self.getTsFromTsId(tsId)
            program = Plugin.getProgram()
            tmpPrefix = self._getTmpPath(tsId)
            inTsFn = self.getFilePath(tsFn, tmpPrefix, tsId, ext=MRC_EXT)
            param = self._genAretomoCmd(ts, inTsFn, tsId)
            self.runJob(program, param, env=Plugin.getEnviron())
        except Exception as e:
            self.failedItems.append(tsId)
            logger.error(redStr(f'tsId = {tsId} -> AreTomo execution failed '
                                f'with the exception -> {e}'))
            logger.error(traceback.format_exc())

    def createOutputStep(self, tsId: str, tsFn: str):
        if tsId in self.failedItems:
            self.createOutputFailedTs(tsId)
            return
        logger.info(cyanStr(f'tsId = {tsId} -> creating the outputs...'))
        self.createOutputs(tsId, tsFn)
        # Close explicitly the outputs (for streaming)
        for outputName in self._possibleOutputs.keys():
            output = getattr(self, outputName, None)
            if output:
                output.close()

    def createOutputs(self, tsId: str, tsFn: str):
        try:
            with self._lock:
                ts = self.getTsFromTsId(tsId, doLock=False)
                doAliTs = self._getAttribVal(DO_ALI_TS)
                doTomoRec = self._getAttribVal(DO_TOMO_REC)
                extraPrefix = self._getExtraPath(tsId)
                AretomoAln = readAlnFile(self.getAlnFile(tsFn, tsId))
                indexDict = self._getIndexAssignDict(ts)
                finalIndsAliDict = {}  # {indexInOrigTs: matching line index in aln (AretomoAln.sections.index(secNum))}
                for secNum, origInd in indexDict.items():
                    if secNum in AretomoAln.sections:
                        finalIndsAliDict[origInd] = AretomoAln.sections.index(secNum)

                finalInds = list(finalIndsAliDict.keys())  # Final enabled indices in the original TS
                alignmentMatrix = getTransformationMatrix(AretomoAln.imod_matrix)

                if doAliTs:
                    # Check the output tilt angles before storing the corresponding outputs
                    inTiltAngles = np.array(
                        [ti.getTiltAngle() for ti in ts if ti.getIndex() in AretomoAln.sections])
                    aretomoTiltAngles = np.array([AretomoAln.tilt_angles])
                    if not np.allclose(inTiltAngles, aretomoTiltAngles, atol=45):
                        msg = 'tsId = %s. Bad tilt angle values detected.' % tsId
                        self.warning(msg + ' Skipping...')
                        outMsg = self.badTsAliMsg.get() + '\n' + msg if self.badTsAliMsg.get() else '\n' + msg
                        self.badTsAliMsg.set(outMsg)
                        self._store(self.badTsAliMsg)
                        return

                if doTomoRec:
                    # Check the tomogram dims before storing the corresponding outputs
                    tomoFileName = self.getFilePath(tsFn, extraPrefix, tsId, suffix='_Vol', ext=MRC_EXT)
                    tomoDims = self._getOutputDim(tomoFileName)
                    if np.any(np.array(tomoDims) == len(ts)):
                        msg = 'tsId = %s. Generated tomogram dims = %s' % (tsId, str(tomoDims))
                        self.warning('Tilt series skipped because of a bad reconstruction. ' + msg)
                        outMsg = self.badTomoRecMsg.get() + '\n' + msg if self.badTomoRecMsg.get() else '\n' + msg
                        self.badTomoRecMsg.set(outMsg)
                        self._store(self.badTomoRecMsg)
                        return
                    outputSetOfTomograms = self.getOutputSetOfTomograms()
                    # Tomogram attributes
                    newTomogram = Tomogram()
                    newTomogram.setLocation(tomoFileName)
                    newTomogram.setSamplingRate(outputSetOfTomograms.getSamplingRate())
                    newTomogram.setOrigin()
                    newTomogram.setAcquisition(ts.getAcquisition())
                    newTomogram.setTsId(tsId)
                    newTomogram.setCtfCorrected(ts.ctfCorrected())
                    if self.doEvenOdd.get():
                        newTomogram.setHalfMaps([self.getFilePath(tsFn, extraPrefix, tsId, suffix=ODD, ext=MRC_EXT),
                                                 self.getFilePath(tsFn, extraPrefix, tsId, suffix=EVEN, ext=MRC_EXT)])
                    outputSetOfTomograms.append(newTomogram)
                    outputSetOfTomograms.update(newTomogram)
                    outputSetOfTomograms.write()
                    self._store(outputSetOfTomograms)
                else:
                    # remove aligned stack from output
                    pwutils.cleanPath(self.getFilePath(tsFn, extraPrefix, tsId, ext=MRC_EXT))

                # Save original TS stack with new alignment,
                # unless making a tomo from pre-aligned TS
                if doAliTs:
                    outputSetOfTiltSeries = self.getOutputSetOfTiltSeries(OUT_TS)
                    newTs = TiltSeries()
                    newTs.copyInfo(ts)
                    newTs.setSamplingRate(self._getInputSampling())
                    newTs.setAlignment2D()
                    outputSetOfTiltSeries.append(newTs)

                    for i, tiltImage in enumerate(ts.iterItems(orderBy=TiltImage.INDEX_FIELD)):
                        newTi = tiltImage.clone()
                        newTi.copyInfo(tiltImage, copyId=True, copyTM=False)
                        transform = Transform()
                        ind = i + 1

                        if ind in finalInds:
                            # Set the tilt angles
                            secIndex = finalIndsAliDict[ind]
                            acq = tiltImage.getAcquisition()
                            newTi.setTiltAngle(AretomoAln.tilt_angles[secIndex])
                            acq.setTiltAxisAngle(AretomoAln.tilt_axes[secIndex])
                            newTi.setAcquisition(acq)

                            # set Transform
                            m = alignmentMatrix[:, :, secIndex]
                            self.debug(
                                f"Section {secNum}: {AretomoAln.tilt_axes[secIndex]}, "
                                f"{AretomoAln.tilt_angles[secIndex]}")
                            transform.setMatrix(m)
                        else:
                            newTi.setEnabled(False)
                            transform.setMatrix(np.identity(3))

                        newTi.setTransform(transform)
                        newTi.setSamplingRate(self._getInputSampling())
                        newTs.append(newTi)

                    # update tilt axis angle for TS with the first value only
                    acq = newTs.getAcquisition()
                    acq.setTiltAxisAngle(AretomoAln.tilt_axes[0])
                    newTs.setAcquisition(acq)

                    newTs.setDim(self._getSetOfTiltSeries().getDim())
                    newTs.write(properties=False)

                    outputSetOfTiltSeries.update(newTs)
                    outputSetOfTiltSeries.write()
                    self._store(outputSetOfTiltSeries)

                    # Output set of CTF tomo series
                    if self._getAttribVal(DO_CTF_EST):
                        outputCtfs = self.getOutputSetOfCtfs()

                        newCTFTomoSeries = CTFTomoSeries()
                        newCTFTomoSeries.copyInfo(newTs)
                        newCTFTomoSeries.setTiltSeries(newTs)
                        newCTFTomoSeries.setTsId(tsId)
                        outputCtfs.append(newCTFTomoSeries)

                        aretomoCtfFile = self.getFilePath(tsFn, extraPrefix, tsId, suffix="CTF", ext=".txt")
                        psdFile = pwutils.replaceExt(aretomoCtfFile, 'mrc')
                        psdFile = psdFile if exists(psdFile) else None
                        ctfResult = AretomoCtfParser.readAretomoCtfOutput(aretomoCtfFile)

                        for i, tiltImage in enumerate(ts.iterItems()):
                            ctf = CTFModel()
                            ind = i + 1
                            if ind in finalInds:
                                secIndex = finalIndsAliDict[ind]
                                AretomoCtfParser._getCtfTi(ctf, ctfResult, item=secIndex, psdFile=psdFile)
                                newCtfTomo = CTFTomo.ctfModelToCtfTomo(ctf)
                            else:
                                ctf.setWrongDefocus()
                                newCtfTomo = CTFTomo.ctfModelToCtfTomo(ctf)
                                newCtfTomo.setEnabled(False)

                            newCtfTomo.setAcquisitionOrder(tiltImage.getAcquisitionOrder())
                            newCTFTomoSeries.append(newCtfTomo)

                        outputCtfs.update(newCTFTomoSeries)
                        outputCtfs.write()
                        self._store(outputCtfs)
        except Exception as e:
            logger.error(redStr(f'tsId = {tsId} -> Unable to register the output with exception {e}. Skipping... '))
            logger.error(traceback.format_exc())

    def createOutputFailedTs(self, tsId: str):
        logger.info(cyanStr(f'Failed TS -> {tsId}'))
        try:
            with self._lock:
                ts = self.getTsFromTsId(tsId, doLock=False)
                inTsSet = self._getSetOfTiltSeries()
                outTsSet = self.getOutputFailedSetOfTiltSeries(inTsSet)
                newTs = TiltSeries()
                newTs.copyInfo(ts)
                outTsSet.append(newTs)
                newTs.copyItems(ts)
                newTs.write()
                outTsSet.update(newTs)
                outTsSet.write()
                self._store(outTsSet)
                # Close explicitly the outputs (for streaming)
                outTsSet.close()
        except Exception as e:
            logger.error(redStr(f'tsId = {tsId} -> Unable to register the failed output with '
                                f'exception {e}. Skipping... '))
            logger.error(traceback.format_exc())

    def closeOutputSetStep(self, attrib: Union[List[str], str]):
        self._closeOutputSet()
        attribList = [attrib] if type(attrib) is str else attrib
        failedOutputList = []
        for attr in attribList:
            outTsSet = getattr(self, attr, None)
            if not outTsSet or (outTsSet and len(outTsSet) == 0):
                failedOutputList.append(attr)
        if failedOutputList:
            raise Exception(f'No output/s {failedOutputList} were generated. Please check the '
                            f'Output Log > run.stdout and run.stderr')

    # --------------------------- INFO functions ------------------------------
    def _summary(self) -> List[str]:
        summary = []
        if hasattr(self, OUT_TOMO):
            summary.append(f"Input tilt-series: "
                           f"{self._getSetOfTiltSeries().getSize()}.\n"
                           f"Tomograms reconstructed: "
                           f"{getattr(self, OUT_TOMO).getSize()}.\n")
        elif hasattr(self, OUT_TS):
            summary.append(f"Input tilt-series: "
                           f"{self._getSetOfTiltSeries().getSize()}.\n"
                           f"Tilt series aligned: "
                           f"{getattr(self, OUT_TS).getSize()}")
        else:
            summary.append("Output is not ready yet.")

        if self.badTsAliMsg.get():
            summary.append('*WARNING!*' + self.badTsAliMsg.get())

        if self.badTomoRecMsg.get():
            summary.append('*WARNING!*\nSome tilt series were skipped because of a bad reconstruction:' +
                           self.badTomoRecMsg.get())

        return summary

    def _validate(self) -> List[str]:
        errors = []
        inTsSet = self._getSetOfTiltSeries()
        self._validateThreads(errors)
        doAliTs = self._getAttribVal(DO_ALI_TS)
        doTomoRec = self._getAttribVal(DO_TOMO_REC)
        aliZ = self._getAttribVal(ALIGN_Z, 0)
        tomoThk = self._getAttribVal(TOMO_THK, 0)
        if self._getAttribVal(DO_CTF_EST) and not doAliTs:
            errors.append('CTF estimation requires the tilt-series alignment to be calculated.')
        if doAliTs and doTomoRec and aliZ > 0 and tomoThk > 0:
            if aliZ >= tomoThk:
                errors.append("Z volume height for alignment should be always "
                              "smaller than tomogram thickness if they both are specified and "
                              "greater than 0.")

        if not doAliTs:
            if doTomoRec:
                if not inTsSet.hasAlignment():
                    errors.append('The tilt-series introduced do not have alignment information.')
            if tomoThk <= 0:
                errors.append('The sample thickness cannot be estimated without aligning the tilt-series.')
            if doTomoRec:
                errors.append("You cannot switch off both alignment and reconstruction.")

        if self.doEvenOdd.get() and not inTsSet.hasOddEven():
            errors.append('The even/odd tomograms cannot be reconstructed as no even/odd tilt-series are found '
                          'in the metadata of the introduced tilt-series.')

        # if self.outImod.get() != 0 and not self.doDW:
        #     errors.append("Dose weighting needs to be enabled when "
        #                   "saving extra IMOD output.")

        return errors

    def _warnings(self) -> List[str]:
        warnMsgs = []
        if self._getSetOfTiltSeries().hasAlignment() and self._getAttribVal(DO_ALI_TS):
            warnMsgs.append("Input tilt-series already have alignment "
                            "information. You probably want to skip the alignment step.")
        return warnMsgs

    # --------------------------- UTILS functions -----------------------------
    def _genAretomoCmd(self,
                       ts: TiltSeries,
                       tsFn: str,
                       tsId: str) -> str:
        acq = ts.getAcquisition()
        extraPrefix = self._getExtraPath(tsId)
        align = 0 if not self._getAttribVal(DO_ALI_TS) else 1
        recTomo = self._getAttribVal(DO_TOMO_REC)
        estimateCtf = self._getAttribVal(DO_CTF_EST)
        aliZ = self._getAttribVal(ALIGN_Z)
        tomoThk = self._getAttribVal(TOMO_THK)

        args = {
            '-InPrefix': tsFn,
            '-OutDir': extraPrefix,
            '-PixSize': ts.getSamplingRate(),  # Pixel size in A of input stack in angstrom.
            '-Kv': acq.getVoltage(),  # High tension in kV needed for dose weighting
            '-SplitSum': 1 if self.doEvenOdd.get() else 0,  # Odd / even management
            '-Align': align,
            '-VolZ': tomoThk if recTomo else 0,  # z height for rec. It must be > 0 to rec a volume
            '-FlipVol': 1 if recTomo and self.flipVol else 0,  # Flip volume from xzy to xyz
            '-FlipInt': 1 if self.flipInt else 0,  # Flip the intensity
            '-DarkTol': self.darkTol.get(),  # Tolerance for removing dark images
            '-OutImod': self.outImod.get(),
            # '-FmDose': self.dosePerRawFrame.get(),
            '-Gpu': '%(GPU)s'
        }

        if align:
            # Volume height for alignment
            # if self.alignZfile.get():
            #     # Check if we have AlignZ information per tilt-series
            #     args['-AlignZ'] = self.perTsAlignZ.get(tsId, self.alignZ)
            # else:
            args['-AlignZ'] = aliZ

        if recTomo:
            # if not align:
            args['-AtBin'] = self.binFactor.get()
            args['-ReconRange'] = f'{self.minTiltRec.get()} {self.maxTiltRec.get()}'
            # args['-InMrc'] = tsFn
            # args['-AlnFile'] = self.getAlnFile(tsFn, tsId)
            if aliZ <= 0 and tomoThk <= 0:
                args['-ExtZ'] = self._getAttribVal(EXTRA_Z_HEIGHT)
            if self.reconMethod == RECON_SART:
                args['-Sart'] = f"{self.SARTiter} {self.SARTproj}"
            else:
                args['-Wbp'] = 1
            

        # CTF estimation
        if estimateCtf:
            args['-Cs'] = acq.getSphericalAberration()
            args['-CorrCTF'] = 1 if self.doCorrCtf.get() else 0
            if self.doPhaseShiftSearch.get():
                args['-ExtPhase'] = f'{self.minPhaseShift} {self.maxPhaseShift}'

        # Tilt-axis management
        tiltAxisAngle = acq.getTiltAxisAngle() or 0.0
        # TODO: check the commented lines below
        # if ts.hasAlignment():
        #     # in this case we already used ts.applyTransform()
        #     tiltAxisAngle = 0.0
        args['-TiltAxis'] = f"{tiltAxisAngle} {self._getTiltAxisOperation()}"

        # Tilt-angles management
        args['-TiltCor'] = self.refineTiltAngles.get() - 1

        # Local alignment management
        if self.doLocalAli.get():
            args['-AtPatch'] = f"{self.patchX} {self.patchY}"

        param = ' '.join([f'{k} {str(v)}' for k, v in args.items()])
        param += ' ' + self.extraParams.get()
        return param

    def readingOutput(self) -> None:
        if not self._getAttribVal(DO_ALI_TS):
            self.__readingOutPutTomos()
        else:
            self.__readingOutPutTsSet()

    def __readingOutPutTomos(self) -> None:
        outTomoSet = getattr(self, OUT_TOMO, None)
        if outTomoSet:
            for ts in outTomoSet:
                self.TS_read.append(ts.getTsId())
            logger.info(cyanStr(f'TsIds processed: {self.TS_read}'))
        else:
            logger.info(cyanStr('No tilt-series have been processed yet'))

    def __readingOutPutTsSet(self) -> None:
        outTsSet = getattr(self, OUT_TS, None)
        if outTsSet:
            for ts in outTsSet:
                self.TS_read.append(ts.getTsId())
            logger.info(cyanStr(f'TsIds processed: {self.TS_read}'))
        else:
            logger.info(cyanStr('No tilt-series have been processed yet'))

    @staticmethod
    def readThicknessFile(filePath: os.PathLike):
        """ Reads a text file with thickness information per tilt-series.
        Example of how the file should look like:
        Position_112 700
        Position_35  650
        Position_18  500
        Position_114 1000
        """
        thickPerTs = {}
        with open(filePath, "r") as f:
            lines = f.readlines()
            lines = filter(lambda x: x.strip(), lines)

            for line in lines:
                values = line.split()
                thickPerTs[values[0]] = values[1]

        return thickPerTs

    @staticmethod
    def getFilePath(tsFn: Union[str, os.PathLike],
                    prefix: str,
                    tsId: str,
                    suffix: Optional[str] = '',
                    ext: Optional[str] = None) -> Union[str, os.PathLike]:
        fileExtension = ext if ext else getExt(tsFn)
        if suffix:
            suffix = suffix if suffix.startswith('_') else '_' + suffix
        return join(prefix, tsId + suffix + fileExtension)

    def getFilePathEven(self,
                        tsFn: Union[str, os.PathLike],
                        prefix: str,
                        tsId: str,
                        ext: str = MRC_EXT) -> Union[str, os.PathLike]:
        return self.getFilePath(tsFn, prefix, tsId, suffix=EVEN, ext=ext)

    def getFilePathOdd(self,
                       tsFn: Union[str, os.PathLike],
                       prefix: str,
                       tsId: str,
                       ext: str = MRC_EXT) -> Union[str, os.PathLike]:
        return self.getFilePath(tsFn, prefix, tsId, suffix=ODD, ext=ext)

    def getOutputSetOfTomograms(self) -> SetOfTomograms:
        outputSetOfTomograms = getattr(self, OUT_TOMO, None)
        if outputSetOfTomograms:
            outputSetOfTomograms.enableAppend()
        else:
            outputSetOfTomograms = self._createSetOfTomograms()
            outputSetOfTomograms.copyInfo(self._getSetOfTiltSeries())
            outputSetOfTomograms.setSamplingRate(self._getOutputSampling())
            outputSetOfTomograms.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(**{OUT_TOMO: outputSetOfTomograms})
            self._defineSourceRelation(self._getSetOfTiltSeries(isPointer=True), outputSetOfTomograms)
        return getattr(self, OUT_TOMO)

    def getOutputSetOfTiltSeries(self,
                                 outputName: str = OUT_TS) -> SetOfTiltSeries:
        outputSetOfTiltSeries = getattr(self, outputName, None)
        if outputSetOfTiltSeries:
            outputSetOfTiltSeries.enableAppend()
        else:
            outputSetOfTiltSeries = self._createSetOfTiltSeries()
            outputSetOfTiltSeries.copyInfo(self._getSetOfTiltSeries())
            pixSize = self._getInputSampling()
            alignment = ALIGN_2D

            # Dimensions will be updated later
            outputSetOfTiltSeries.setDim(self._getSetOfTiltSeries().getDim())
            outputSetOfTiltSeries.setSamplingRate(pixSize)
            outputSetOfTiltSeries.setAlignment(alignment)
            outputSetOfTiltSeries.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(**{outputName: outputSetOfTiltSeries})
            self._defineSourceRelation(self._getSetOfTiltSeries(isPointer=True),
                                       outputSetOfTiltSeries)
        return outputSetOfTiltSeries

    def getOutputSetOfCtfs(self) -> SetOfCTFTomoSeries:
        outputCtfs = getattr(self, OUT_CTFS, None)
        if outputCtfs:
            outputCtfs.enableAppend()
        else:
            inTsPointer = self._getSetOfTiltSeries(isPointer=True)
            outputCtfs = SetOfCTFTomoSeries.create(self._getPath(), template='CTFmodels%s.sqlite')
            outputCtfs.setSetOfTiltSeries(inTsPointer)
            outputCtfs.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(**{OUT_CTFS: outputCtfs})
            self._defineSourceRelation(inTsPointer, outputCtfs)
        return outputCtfs

    def _getSetOfTiltSeries(self, isPointer: bool = False) -> Union[Pointer, SetOfTiltSeries]:
        if isPointer:
            return self.inputSetOfTiltSeries
        else:
            return self.inputSetOfTiltSeries.get()

    def _getOutputSampling(self) -> float:
        return self._getInputSampling() * self.binFactor.get()

    def _getInputSampling(self) -> float:
        return self._getSetOfTiltSeries().getSamplingRate()

    @staticmethod
    def _getOutputDim(fn: str) -> Tuple[int, int, int]:
        ih = ImageHandler()
        x, y, z, _ = ih.getDimensions(fn)
        return x, y, z

    def getOutputFailedSetOfTiltSeries(self, inputSet):
        failedTsSet = getattr(self, FAILED_TS, None)
        if failedTsSet:
            failedTsSet.enableAppend()
        else:
            failedTsSet = SetOfTiltSeries.create(self._getPath(), template='tiltseries', suffix='Failed')
            failedTsSet.copyInfo(inputSet)
            failedTsSet.setDim(inputSet.getDim())
            failedTsSet.setStreamState(Set.STREAM_OPEN)

            self._defineOutputs(**{FAILED_TS: failedTsSet})
            self._defineSourceRelation(self._getSetOfTiltSeries(isPointer=True), failedTsSet)

        return failedTsSet

    def getTsFromTsId(self,
                      tsId: str,
                      doLock: bool = True) -> TiltSeries:
        tsSet = self._getSetOfTiltSeries()
        if doLock:
            with self._lock:
                return tsSet.getItem(TiltSeries.TS_ID_FIELD, tsId)
        else:
            return tsSet.getItem(TiltSeries.TS_ID_FIELD, tsId)

    def getAlnFile(self, tsFn: str, tsId: str):
        return self.getFilePath(tsFn, self._getExtraPath(tsId), tsId, ext=".aln")

    @staticmethod
    def _getIndexAssignDict(ts: TiltSeries) -> dict:
        """It generates a dictionary of {key: value} = {indInRestackedTs: indInOriginalTs} that will
        be used to get the excluded views after the re-stacking process in case there are excluded views in
        the input tilt-series (so they are re-stacked in the convert input step) and the automatically excluded
        views because of the dark tolerance threshold of AreTomo, that are excluded considering the indices of
        the re-stacked tilt-series, but the output non-interpolated tilt-series must be referred to the
        input tilt-series."""
        indexDict = {}
        newInd = 1
        for ti in ts.iterItems():
            if ti.isEnabled():
                indexDict[newInd] = ti.getIndex()
                newInd += 1
        return indexDict

    def _getOutputsToCheck(self) -> List[str]:
        outputsToCheck = []
        doAlignTs = self._getAttribVal(DO_ALI_TS)
        if doAlignTs:
            outputsToCheck.append(OUT_TS)
        if self._getAttribVal(DO_TOMO_REC):
            outputsToCheck.append(OUT_TOMO)
        if self._getAttribVal(DO_CTF_EST) and doAlignTs:
            outputsToCheck.append(OUT_CTFS)
        return outputsToCheck

    def _getTiltAxisOperation(self):
        """Get the value of the user selected operation for the tilt-axis angle refinement.
        Extracted from the manual of AreTomo3:

        -TiltAxis can be used to pass a user specified angle of tilt axis into AreTomo3.
        It can take two parameters of which the first one is the user provided angle of
        tilt axis. The second one is optional. The default value is 0, which lets AreTomo3
        refine the angle of tilt axis within +/-10 deg. If the second number is positive, the
        refinement is done within +/-3 deg. When the second number is negative, AreTomo3
        uses the user provided value, i.e. the first number, without any refinement.
        """
        doRefineTiltAxis = self.refineTiltAxis.get()
        if doRefineTiltAxis == NO_REFINE:
            return -1
        elif doRefineTiltAxis == REFINE_10_DEG:
            return 0
        else:  # Refine 3 degrees
            return 1

    @staticmethod
    def _genAretomo3TltFile(ts: TiltSeries,
                            tltFilePath: str,
                            reverse: bool = False,
                            presentAcqOrders: typing.Set[int] = None,
                            includeDose: bool = False) -> None:
        """ Generates an angle file in .tlt format in the specified location. If reverse is set to true the angles in
        file are sorted in the opposite order. AreTomo3 also looks for the corresponding tilt angle file that has the
        same file name except a different file extension. The extension of tilt angle files must be either .rawtlt or
        _TLT.txt. Tilt series files and the associated tilt angle files must be in the same flat directory. In a tilt
        angle file, column 1 in the angle file is mandatory, which contains the tilt angles in the same order as in
        the tilt series. Column 2 is optional and the order of the acquisition. Column 3 is the image dose in
        electrons / squared angstroms.

        :param tltFilePath: String containing the path where the file is created.
        :param reverse: Boolean indicating if the angle list must be reversed.
        :param presentAcqOrders: set containing the present acq orders in both the given TS and CTFTomoSeries. Used to
        filter the tilt angles that will be written in the tlt file generated. The parameter excludedViews is ignored
        if presentAcqOrders is provided, as the excluded views info may have been used to generate the presentAcqOrders
        (see tomo > utils > getCommonTsAndCtfElements)
        :param includeDose: boolean used to indicate if the tlt file created must contain an additional column with
        the dose or not (default).
        """
        angleList = []
        acqOrderList = []
        doseList = []
        if presentAcqOrders:
            for ti in ts.iterItems(orderBy=TiltImage.TILT_ANGLE_FIELD):
                acqOrder = ti.getAcquisitionOrder()
                if acqOrder in presentAcqOrders:
                    angleList.append(ti.getTiltAngle())
                    acqOrderList.append(acqOrder)
                    if includeDose:
                        doseList.append(ti.getAcquisition().getAccumDose())
        else:
            for ti in ts.iterItems(orderBy=TiltImage.TILT_ANGLE_FIELD):
                angleList.append(ti.getTiltAngle())
                acqOrderList.append(ti.getAcquisitionOrder())
                if includeDose:
                    doseList.append(ti.getAcquisition().getAccumDose())

        if reverse:
            angleList.reverse()
            acqOrderList.reverse()
            if includeDose:
                doseList.reverse()

        with open(tltFilePath, 'w') as f:
            if includeDose:
                f.writelines(f"{angle:0.3f}\t{acqOrder}\t{dose:0.4f}\n" for angle, acqOrder, dose in
                             zip(angleList, acqOrderList, doseList))
            else:
                f.writelines(f"{angle:0.3f}\t{acqOrder} \n" for angle, acqOrder in zip(angleList, acqOrderList))

    def _getAttribVal(self, attribName: str, defaultVal: Union[int, float, bool, None] = None) \
            -> Union[int, float, bool, None]:
        attribVal = getattr(self, attribName, None)
        return attribVal if attribVal else defaultVal
