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

import os
import numpy as np
import time
from typing import List, Literal, Tuple, Union, Optional
from pwem import ALIGN_NONE, ALIGN_2D
from pyworkflow.protocol import params, STEPS_PARALLEL
from pyworkflow.constants import PROD
from pyworkflow.object import Set, String, Pointer
from pyworkflow.protocol import ProtStreamingBase
import pyworkflow.utils as pwutils
from pwem.protocols import EMProtocol
from pwem.objects import Transform, CTFModel
from pwem.emlib.image import ImageHandler
from pyworkflow.utils import Message, cyanStr, removeBaseExt, getExt, createLink
from tomo.protocols import ProtTomoBase
from tomo.objects import (Tomogram, TiltSeries, TiltImage,
                          SetOfTomograms, SetOfTiltSeries, SetOfCTFTomoSeries, CTFTomoSeries, CTFTomo)

from .. import Plugin
from ..convert.convert import getTransformationMatrix, readAlnFile, writeAlnFile
from ..convert.dataimport import AretomoCtfParser
from ..constants import RECON_SART, LOCAL_MOTION_COORDS, LOCAL_MOTION_PATCHES

OUT_TS = "TiltSeries"
OUT_TS_ALN = "InterpolatedTiltSeries"
OUT_TOMO = "Tomograms"
OUT_CTFS = "CTFTomoSeries"
FAILED_TS = 'FailedTiltSeries'
EVEN = '_even'
ODD = '_odd'
MRC_EXT = '.mrc'


class ProtAreTomoAlignRecon(EMProtocol, ProtTomoBase, ProtStreamingBase):
    """ Protocol for fiducial-free alignment and reconstruction for tomography available in streaming. """
    _label = 'tilt-series align and reconstruct'
    _devStatus = PROD
    _possibleOutputs = {OUT_TS: SetOfTiltSeries,
                        OUT_TS_ALN: SetOfTiltSeries,
                        FAILED_TS: SetOfTiltSeries,
                        OUT_TOMO: SetOfTomograms,
                        OUT_CTFS: SetOfCTFTomoSeries}
    stepsExecutionMode = STEPS_PARALLEL

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)
        self.TS_read = []
        self.badTsAliMsg = String()
        self.badTomoRecMsg = String()
        self.excludedViewsMsg = String()
        self._failedTsList = []

        # --------------------------- DEFINE param functions ----------------------

    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam('inputSetOfTiltSeries',
                      params.PointerParam,
                      pointerClass='SetOfTiltSeries',
                      important=True,
                      label='Input set of Tilt-Series',
                      help='If you choose to skip alignment, the input '
                           'tilt-series are expected to be already aligned.')

        form.addParam('skipAlign', params.BooleanParam,
                      default=False,
                      label='Skip alignment?',
                      help='You can skip alignment if you just want to '
                           'reconstruct a tomogram from already '
                           'aligned tilt-series.')

        form.addParam('makeTomo', params.BooleanParam,
                      default=True,
                      label='Reconstruct the tomograms?',
                      help='You can skip tomogram reconstruction, so that input '
                           'tilt-series will be only aligned.')

        form.addParam('doEvenOdd', params.BooleanParam,
                      label='Reconstruct the odd/even tomograms?',
                      default=False,
                      condition='makeTomo')

        doAlignTs = 'not skipAlign'
        form.addParam('saveStack', params.BooleanParam,
                      condition="not makeTomo and not skipAlign",
                      default=True,
                      label="Save interpolated aligned TS?",
                      help="Choose No to discard aligned stacks.")

        form.addParam('binFactor', params.IntParam,
                      default=2,
                      label='Binning',
                      important=True,
                      help='Binning for aligned output tilt-series / volume.')

        form.addParam('alignZ', params.IntParam, default=800,
                      condition=doAlignTs,
                      important=True,
                      label='Volume height for alignment (voxels)',
                      help='Specifies Z height (*unbinned*) of the temporary volume '
                           'reconstructed for projection matching as part '
                           'of the alignment process. This value plays '
                           'an important role in alignment accuracy. This '
                           'Z height should be always smaller than tomogram '
                           'thickness and should be close to the sample '
                           'thickness.')

        form.addParam('alignZfile', params.FileParam,
                      expertLevel=params.LEVEL_ADVANCED,
                      condition=doAlignTs,
                      label='File with volume height for alignment per tilt-series',
                      help='Specifies a text file containing the Z height (*unbinned*) '
                           'to be used for alignment of individual tilt-series. '
                           'The file should have two columns, the first '
                           'containing the tsId and the second containing the AlignZ value '
                           'for that tilt-series. You can specify one tilt-series '
                           'per line.')

        form.addParam('tomoThickness', params.IntParam,
                      condition='makeTomo',
                      important=True,
                      default=1200,
                      label='Tomogram thickness unbinned (voxels)',
                      help='Z height of the reconstructed volume in '
                           '*unbinned* voxels.')

        form.addParam('refineTiltAngles',
                      params.EnumParam,
                      condition=doAlignTs,
                      choices=['No', 'Measure only', 'Measure and correct'],
                      display=params.EnumParam.DISPLAY_COMBO,
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

        form.addParam('refineTiltAxis', params.EnumParam,
                      condition=doAlignTs,
                      choices=['No',
                               'Refine and use the refined value for the entire tilt series',
                               'Refine and calculate tilt axis at each tilt angle'],
                      display=params.EnumParam.DISPLAY_COMBO,
                      label="Refine tilt axis angle?",
                      default=1,
                      help="Tilt axis determination is a two-step processing in AreTomo. "
                           "A single tilt axis is first calculated followed by the determination "
                           "of how tilt axis varies over the entire tilt range. The initial "
                           "value lets users enter their estimate and AreTomo refines the "
                           "estimate in [-3º, 3º] range.")

        form.addParam('outImod', params.EnumParam,
                      display=params.EnumParam.DISPLAY_COMBO,
                      condition=doAlignTs,
                      expertLevel=params.LEVEL_ADVANCED,
                      choices=['No', 'Relion 4', 'Warp', 'Save locally aligned TS'],
                      default=0,
                      label="Generate extra IMOD output?",
                      help="0 - No\n1 - generate IMOD files for Relion 4\n"
                           "2 - generate IMOD files for Warp\n"
                           "3 - generate global and local-aligned tilt series stack. "
                           "High frequencies are enhanced to alleviate the attenuation "
                           "due to interpolation.")

        group = form.addGroup('CTF', condition=doAlignTs)
        group.addParam('doEstimateCtf', params.BooleanParam,
                       default=True,
                       label='Estimate the CTF?',
                       condition=doAlignTs)

        group.addParam('doPhaseShiftSearch', params.BooleanParam,
                       default=False,
                       label='Do phase shift estimation?',
                       condition='doEstimateCtf')
        linePhaseShift = group.addLine('Phase shift range (deg.)',
                                       condition='doPhaseShiftSearch',
                                       help="Search range of the phase shift (start, end).")
        linePhaseShift.addParam('minPhaseShift', params.IntParam,
                                default=0,
                                label='min',
                                condition='doPhaseShiftSearch')
        linePhaseShift.addParam('maxPhaseShift', params.IntParam,
                                default=0,
                                label='max',
                                condition='doPhaseShiftSearch')

        form.addSection(label='Extra options')
        form.addParam('doDW', params.BooleanParam,
                      default=False,
                      label="Do dose-weighting?")
        form.addParam('reconMethod', params.EnumParam,
                      choices=['SART', 'WBP'],
                      default=RECON_SART,
                      condition='makeTomo',
                      display=params.EnumParam.DISPLAY_HLIST,
                      label="Reconstruction method",
                      help="Choose either SART or weighted back "
                           "projection (WBP).")

        line = form.addLine("SART options", condition=f'reconMethod=={RECON_SART}')
        line.addParam('SARTiter', params.IntParam,
                      default=15,
                      label='iterations')
        line.addParam('SARTproj', params.IntParam,
                      default=5,
                      label='projections per subset')

        form.addParam('flipInt', params.BooleanParam,
                      default=False,
                      label="Flip intensity?",
                      help="By default, the reconstructed volume "
                           "and the input tilt series use the same grayscale "
                           "that makes dense structures dark.")

        form.addParam('flipVol', params.BooleanParam,
                      condition="makeTomo",
                      default=True,
                      label="Flip volume?",
                      help="Set to Yes when making a tomogram and No when "
                           "making a tilt-series. This way the output orientation "
                           "will be similar to IMOD.")

        form.addParam('roiArea', params.StringParam, default='',
                      condition=doAlignTs,
                      expertLevel=params.LEVEL_ADVANCED,
                      label="ROI for focused alignment",
                      help="By default AreTomo assumes the region of interest "
                           "at the center of 0º projection image. A circular "
                           "mask is employed to down-weight the area outside "
                           "ROI during the alignment. When the structures of "
                           "interest are far away from the tilt axis, the "
                           "angular error in the determination of tilt axis "
                           "will significantly amplify the translational error. "
                           "ROI function can effectively improve the alignment "
                           "accuracy for the distant structures.\nHere you can "
                           "provide *a pair of x and y coordinates*, representing "
                           "the center of the region of interest.\n"
                           "The region of interest should be selected from 0º "
                           "projection image with the origin at the lower left "
                           "corner. IMOD's Pixel View is a good tool to select "
                           "the center of region of interest.")

        group = form.addGroup('Local motion correction')
        group.addParam('sampleType', params.EnumParam,
                       # condition="not useInputProt",
                       choices=['Disable local correction', 'Isolated',
                                'Well distributed'],
                       display=params.EnumParam.DISPLAY_COMBO,
                       label="Sample type",
                       default=0,
                       help="AreTomo provides two means to correct the local "
                            "motion, one for isolated sample and the other "
                            "for well distributed across the field of view.")

        group.addParam('coordsFn', params.FileParam,
                       default='',
                       label='Coordinate file',
                       condition='sampleType==%d' % LOCAL_MOTION_COORDS,
                       help="A list of x and y coordinates should be put "
                            "into a two-column text file, one column for x "
                            "and the other for y. Each pair defines a region "
                            "of interest (ROI). The origin of the coordinate "
                            "system is at the image's lower left corner.")

        line = group.addLine("Patches", condition='sampleType==%d' % LOCAL_MOTION_PATCHES)
        line.addParam('patchX', params.IntParam,
                      default=5,
                      label='X')
        line.addParam('patchY', params.IntParam,
                      default=5,
                      label='Y')

        form.addParam('darkTol', params.FloatParam,
                      default=0.7,
                      condition=doAlignTs,
                      important=True,
                      label="Dark tolerance",
                      help="Set tolerance for removing dark images. The range is "
                           "in (0, 1). The default value is 0.7. "
                           "The higher value is more restrictive.")

        form.addParam('extraParams', params.StringParam,
                      default='',
                      label='Additional parameters',
                      help="Extra command line parameters. See AreTomo help.")

        form.addHidden(params.GPU_LIST, params.StringParam,
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

        while True:
            listTSInput = inTsSet.getTSIds()
            if not inTsSet.isStreamOpen() and self.TS_read == listTSInput:
                self.info(cyanStr('Input set closed, all items processed\n'))
                self._insertFunctionStep(self.closeOutputSetStep,
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
                    self.info(cyanStr(f"Steps created for TS_ID: {tsId}"))
                    self.TS_read.append(tsId)

            time.sleep(10)
            if inTsSet.isStreamOpen():
                with self._lock:
                    inTsSet.loadAllProperties() # refresh status for the streaming

    # --------------------------- STEPS functions -----------------------------
    def convertInputStep(self, tsId: str, tsFn: str):
        self.info(cyanStr(f'------- convertInputStep ts_id: {tsId}'))
        ts = self.getTsFromTsId(tsId)
        presentAcqOrders = ts.getTsPresentAcqOrders()

        extraPrefix = self._getExtraPath(tsId)
        tmpPrefix = self._getTmpPath(tsId)
        pwutils.makePath(*[tmpPrefix, extraPrefix])
        outputTsFileName = self.getFilePath(tsFn, tmpPrefix, tsId, ext=MRC_EXT)

        if self.skipAlign:
            if self.makeTomo:
                createLink(tsFn, outputTsFileName)
                alnFile = self.getAlnFile(tsFn, tsId)
                writeAlnFile(ts, tsFn, alnFile)
        else:
            # Apply the transformation for the input tilt-series
            rotationAngle = ts.getAcquisition().getTiltAxisAngle()
            doSwap = 45 < abs(rotationAngle) < 135
            if self.doEvenOdd.get():
                outputTsFnEven = self.getFilePathEven(tsFn, tmpPrefix, tsId, ext=MRC_EXT)
                outputTsFnOdd = self.getFilePathOdd(tsFn, tmpPrefix, tsId, ext=MRC_EXT)
                ts.applyTransformToAll(outputTsFileName,
                                       swapXY=doSwap,
                                       presentAcqOrders=presentAcqOrders,
                                       outFileNamesEvenOdd=[outputTsFnEven, outputTsFnOdd])
            else:
                ts.applyTransform(outputTsFileName,
                                  swapXY=doSwap,
                                  presentAcqOrders=presentAcqOrders)

            # Generate angle file
            angleFilePath = self.getFilePath(tsFn, tmpPrefix, tsId, ext=".tlt")
            ts.generateTltFile(angleFilePath,
                               presentAcqOrders=presentAcqOrders,
                               includeDose=self.doDW.get())

            if self.alignZfile.hasValue():
                alignZfile = self.alignZfile.get()
                if os.path.exists(alignZfile):
                    self.perTsAlignZ = self.readThicknessFile(alignZfile)

    def runAreTomoStep(self, tsId: str, tsFn: str):
        """ Call AreTomo with the appropriate parameters. """
        self.info(cyanStr(f'------- runAreTomoStep ts_id: {tsId}'))
        try:
            ts = self.getTsFromTsId(tsId)
            program = Plugin.getProgram()
            tmpPrefix = self._getTmpPath(tsId)
            inTsFn = self.getFilePath(tsFn, tmpPrefix, tsId, ext=MRC_EXT)
            param = self._genAretomoCmd(ts, inTsFn, tsId)
            self.runJob(program, param, env=Plugin.getEnviron())
            if self.doEvenOdd.get():
                tmpPrefix = self._getTmpPath(tsId)
                inTsFnOdd = self.getFilePathOdd(tsFn, tmpPrefix, tsId, ext=MRC_EXT)
                inTsFnEven = self.getFilePathEven(tsFn, tmpPrefix, tsId, ext=MRC_EXT)
                # Odd
                self.info(cyanStr(f'------- runAreTomoStep ts_id: {tsId} ODD'))
                param = self._genAretomoCmd(ts, inTsFnOdd, tsId, even=False)
                self.runJob(program, param, env=Plugin.getEnviron())
                # Even
                self.info(cyanStr(f'------- runAreTomoStep ts_id: {tsId} EVEN'))
                param = self._genAretomoCmd(ts, inTsFnEven, tsId, even=True)
                self.runJob(program, param, env=Plugin.getEnviron())

        except Exception as e:
            self._failedTsList.append(tsId)
            self.error('Aretomo execution failed for tsId %s -> %s' % (tsId, e))

    def createOutputStep(self, tsId: str, tsFn: str):
        with self._lock:
            if tsId in self._failedTsList:
                self.createOutputFailedTs(tsId)
                failedTsSet = getattr(self, FAILED_TS, None)
                if failedTsSet:
                    failedTsSet.close()
            else:
                self.createOutputTs(tsId, tsFn)
            for outputName in self._possibleOutputs.keys():
                output = getattr(self, outputName, None)
                if output:
                    output.close()

    def createOutputTs(self, tsId: str, tsFn: str):
        self.info(cyanStr(f'------- createOutputStep ts_id: {tsId}'))

        ts = self.getTsFromTsId(tsId, doLock=False)
        extraPrefix = self._getExtraPath(tsId)
        AretomoAln = readAlnFile(self.getAlnFile(tsFn, tsId))
        indexDict = self._getIndexAssignDict(ts)
        finalIndsAliDict = {}  # {indexInOrigTs: matching line index in aln (AretomoAln.sections.index(secNum))}
        for newInd, origInd in indexDict.items():
            secNum = newInd - 1  # Indices begin in 1, sects in 0
            if secNum in AretomoAln.sections:
                finalIndsAliDict[origInd] = AretomoAln.sections.index(secNum)

        finalInds = list(finalIndsAliDict.keys())  # Final enabled indices in the original TS
        alignmentMatrix = getTransformationMatrix(AretomoAln.imod_matrix)

        if not (self.makeTomo and self.skipAlign):
            # We found the following behavior to be happening sometimes (non-systematically):
            # It can be observed that the tilt angles are badly set for the non-excluded views:
            #
            # AreTomo Alignment / Priims bprmMn
            # RawSize = 512 512 61
            # NumPatches = 0
            # DarkFrame =     0    0   -55.00
            # DarkFrame =     1    1   -53.00
            # DarkFrame =     2    2   -51.00
            # DarkFrame =     3    3   -49.00
            # DarkFrame =     4    4   -47.00
            # DarkFrame =     5    5   -45.00
            # DarkFrame =     6    6   -43.00
            # DarkFrame =    58   58    61.00
            # DarkFrame =    59   59    63.00
            # DarkFrame =    60   60    65.00
            # SEC     ROT         GMAG       TX          TY      SMEAN     SFIT    SCALE     BASE     TILT
            #     7   -10.6414    1.00000     30.409     -7.146     1.00     1.00     1.00     0.00  1567301525373690323140608.00
            #     8   -10.6414    1.00000     25.102     -4.066     1.00     1.00     1.00     0.00  1567301525373690323140608.00
            #     9   -10.6414    1.00000     28.247     -7.649     1.00     1.00     1.00     0.00  1567301525373690323140608.00
            #    10   -10.6414    1.00000     24.338     -5.868     1.00     1.00     1.00     0.00  1567301525373690323140608.00
            #
            # Hence, the output tilt angles will be checked before storing the corresponding outputs
            inTiltAngles = np.array([ti.getTiltAngle() for ti in ts if ti.getIndex() - 1 in AretomoAln.sections])
            aretomoTiltAngles = np.array([AretomoAln.tilt_angles])
            if not np.allclose(inTiltAngles, aretomoTiltAngles, atol=45):
                msg = 'tsId = %s. Bad tilt angle values detected.' % tsId
                self.warning(msg + ' Skipping...')
                outMsg = self.badTsAliMsg.get() + '\n' + msg if self.badTsAliMsg.get() else '\n' + msg
                self.badTsAliMsg.set(outMsg)
                self._store(self.badTsAliMsg)
                return

        if self.makeTomo:
            # Some combinations of the graphic card and cuda toolkit seem to be unstable. Aretomo devs think it may be
            # related to graphic cards with a compute capability greater than 8.6. The behavior observed is detailed
            # below:
            #
            # The non-systematic behavior reported is based on the fact that the dimensions of the tomograms
            # reconstructed (bin 4) are:
            #
            # Sometimes both are well --> dimensions: 958 x 926 x 300
            # Sometimes both are wrong --> dimensions: 958 x no.TiltImages x 926
            # Sometimes one is well and the other wrong, changing the one which is well and the one which is wrong
            # across multiple executions.
            #
            # Until it's clarified, we'll check the dimensions of the generated tomogram and avoid storing the
            # corresponding results if it was badly generated (consequence of a bad alignment with weird tilt angle
            # values, see comment above).
            #
            # Hence, the output tilt angles will be checked before storing the corresponding outputs
            tomoFileName = self.getFilePath(tsFn, extraPrefix, tsId, ext=MRC_EXT)
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
            if self._saveInterpolated():
                # Create new set of aligned TS with potentially fewer tilts included
                outTsAligned = self.getOutputSetOfTiltSeries(OUT_TS_ALN)
                newTs = TiltSeries(tsId=tsId)
                newTs.copyInfo(ts)
                newTs.setSamplingRate(self._getOutputSampling())
                outTsAligned.append(newTs)

                excludedViewsList = []
                accumDoseList = []
                initialDoseList = []
                tiltAngleList = []
                for i, tiltImage in enumerate(ts.iterItems(orderBy=TiltImage.INDEX_FIELD)):
                    ind = i + 1

                    if ind in finalInds:
                        newTi = TiltImage()
                        newTi.copyInfo(tiltImage, copyId=False, copyTM=False)
                        acqTi = tiltImage.getAcquisition()
                        acqTi.setTiltAxisAngle(0.)

                        secIndex = finalIndsAliDict[ind]
                        tiltAngle = AretomoAln.tilt_angles[secIndex]
                        tiltAngleList.append(tiltAngle)
                        newTi.setTiltAngle(tiltAngle)
                        newTi.setLocation(secIndex + 1,
                                          (self.getFilePath(tsFn, extraPrefix, tsId, ext=MRC_EXT)))
                        newTi.setSamplingRate(self._getOutputSampling())
                        newTs.append(newTi)
                        # If the interpolated TS was generated considering the dose weighting, it's accumulated dose
                        # is set to 0 to avoid double dose correction if using the interp TS for the PPPT
                        if self.doDW.get():
                            acqTi.setDoseInitial(0.)
                            acqTi.setAccumDose(0.)
                            acqTi.setDosePerFrame(0.)
                            newTi.setAcquisition(acqTi)
                        else:
                            initialDoseList.append(acqTi.getDoseInitial())
                            accumDoseList.append(acqTi.getAccumDose())

                    else:
                        excludedViewsList.append(ind)
                if excludedViewsList:
                    newTs.setAnglesCount(len(newTs))
                    prevMsg = self.excludedViewsMsg.get() if self.excludedViewsMsg.get() else ''
                    newMsg = f'\n{tsId}: {excludedViewsList}'
                    self.warning('Some views were excluded:' + prevMsg)
                    self.excludedViewsMsg.set(prevMsg + newMsg)
                    self._store(self.excludedViewsMsg)

                acq = newTs.getAcquisition()
                if self.doDW.get():
                    acq.setDoseInitial(0.)
                    acq.setAccumDose(0.)
                    acq.setDosePerFrame(0.)
                else:
                    # The interp TS initial and accumulated dose values may need to be updated in the interpolated
                    # TS if DW is not applied and there are excluded views
                    acq.setAccumDose(max(accumDoseList))
                    acq.setDoseInitial(min(initialDoseList))

                acq.setTiltAxisAngle(0.)  # 0 because TS is aligned
                acq.setAngleMin(min(tiltAngleList))
                acq.setAngleMax(max(tiltAngleList))
                newTs.setAcquisition(acq)

                dims = self._getOutputDim(newTi.getFileName())
                newTs.setInterpolated(True)
                newTs.setDim(dims)
                newTs.write(properties=False)

                outTsAligned.update(newTs)
                outTsAligned.write()
                self._store(outTsAligned)
            else:
                # remove aligned stack from output
                pwutils.cleanPath(self.getFilePath(tsFn, extraPrefix, tsId, ext=MRC_EXT))

        # Save original TS stack with new alignment,
        # unless making a tomo from pre-aligned TS
        if not (self.makeTomo and self.skipAlign):
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
            if self.doEstimateCtf:
                outputCtfs = self.getOutputSetOfCtfs()

                newCTFTomoSeries = CTFTomoSeries()
                newCTFTomoSeries.copyInfo(newTs)
                newCTFTomoSeries.setTiltSeries(newTs)
                newCTFTomoSeries.setTsId(tsId)
                outputCtfs.append(newCTFTomoSeries)

                aretomoCtfFile = self.getFilePath(tsFn, extraPrefix, tsId, suffix="ctf", ext=".txt")
                psdFile = pwutils.replaceExt(aretomoCtfFile, 'mrc')
                psdFile = psdFile if os.path.exists(psdFile) else None
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

    def createOutputFailedTs(self, tsId: str):
        ts = self.getTsFromTsId(tsId, doLock=False)
        inTsSet = self._getSetOfTiltSeries()
        outTsSet = self.getOutputFailedSetOfTiltSeries(inTsSet)
        newTs = TiltSeries()
        newTs.copyInfo(ts)
        outTsSet.append(newTs)
        newTs.copyItems(ts)
        newTs.write(properties=False)
        outTsSet.update(newTs)
        outTsSet.write()
        self._store(outTsSet)

    def closeOutputSetStep(self):
        self._closeOutputSet()
        if self.makeTomo:
            outSet = getattr(self, OUT_TOMO, [])
        else:
            outSet = getattr(self, OUT_TS, [])
        if len(outSet) == 0:
            msg = ('All the introduced Tilt Series were not correctly aligned and/or all the corresponding '
                   'tomograms were not correctly generated.')
            raise Exception(msg)

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

        if self._saveInterpolated() and not self.makeTomo and self.excludedViewsMsg.get():
            summary.append("*Interpolated TS stacks have a few tilt images removed (could be because "
                           "they were marked in the input tilt-series as disabled or because of "
                           "AreTomo's dark tolerance threshold).*\n" +
                           self.excludedViewsMsg.get())

        return summary

    def _validate(self) -> List[str]:
        errors = []
        inTsSet = self._getSetOfTiltSeries()
        self._validateThreads(errors)
        if not self.skipAlign and self.makeTomo and self.alignZ >= self.tomoThickness:
            errors.append("Z volume height for alignment should be always "
                          "smaller than tomogram thickness.")

        if self.skipAlign and self.makeTomo and not inTsSet.hasAlignment():
            errors.append('The tilt-series introduced do not have alignment information.')

        if self.doEvenOdd.get() and not inTsSet.hasOddEven():
            errors.append('The even/odd tomograms cannot be reconstructed as no even/odd tilt-series are found '
                          'in the metadata of the introduced tilt-series.')

        if self.skipAlign and not self.makeTomo:
            errors.append("You cannot switch off both alignment and reconstruction.")

        if self.outImod.get() != 0 and not self.doDW:
            errors.append("Dose weighting needs to be enabled when "
                          "saving extra IMOD output.")

        return errors

    def _warnings(self) -> List[str]:
        warnMsgs = []
        if self._getSetOfTiltSeries().hasAlignment() and not self.skipAlign:
            warnMsgs.append("Input tilt-series already have alignment "
                          "information. You probably want to skip the alignment step.")
        return warnMsgs


    # --------------------------- UTILS functions -----------------------------
    def _genAretomoCmd(self,
                       ts: TiltSeries,
                       tsFn: str,
                       tsId: str,
                       even: Union[bool, None] = None) -> str:
        acq = ts.getAcquisition()

        extraPrefix = self._getExtraPath(tsId)
        tmpPrefix = self._getTmpPath(tsId)
        align = 0
        recTomo= True
        estimateCtf = False
        if even is None:
            outFile = self.getFilePath(tsFn, extraPrefix, tsId, ext=MRC_EXT)
            align = 0 if self.skipAlign else 1
            recTomo = self.makeTomo
            estimateCtf = self.doEstimateCtf.get()
        elif even:
            outFile = self.getFilePath(tsFn, extraPrefix, tsId, suffix=EVEN, ext=MRC_EXT)
        else:
            outFile = self.getFilePath(tsFn, extraPrefix, tsId, suffix=ODD, ext=MRC_EXT)

        args = {
            '-InMrc': tsFn,
            '-OutMrc': outFile,
            '-OutImod': self.outImod.get(),
            '-Align': align,
            '-VolZ': self.tomoThickness if recTomo else 0,
            '-OutBin': self.binFactor,
            '-FlipInt': 1 if self.flipInt else 0,
            '-FlipVol': 1 if recTomo and self.flipVol else 0,
            '-PixSize': ts.getSamplingRate(),
            '-Kv': acq.getVoltage(),
            '-DarkTol': self.darkTol.get(),
            '-AmpContrast': acq.getAmplitudeContrast(),
            '-Gpu': '%(GPU)s'
        }
        if self.doDW:
            args['-ImgDose'] = acq.getDosePerFrame()

        if align:
            args['-AngFile'] = self.getFilePath(tsFn, tmpPrefix, tsId, ext=".tlt")
            if self.alignZfile.get():
                # Check if we have AlignZ information per tilt-series
                args['-AlignZ'] = self.perTsAlignZ.get(tsId, self.alignZ)
            else:
                args['-AlignZ'] = self.alignZ

        if recTomo:
            if not align:
                args['-InMrc'] = tsFn
                args['-AlnFile'] = self.getAlnFile(tsFn, tsId)
            if self.reconMethod == RECON_SART:
                args['-Sart'] = f"{self.SARTiter} {self.SARTproj}"
            else:
                args['-Wbp'] = 1

        if estimateCtf:
            # Manage the CTF estimation:
            # In AreTomo2, parameters PixSize, Kv and Cs are required to estimate the CTF. Since the first two are
            # also used for the dose weighting and the third is only used for the CTF estimation, we'll use it as
            # doEstimateCtf flag parameter.
            args['-Cs'] = acq.getSphericalAberration()
            if self.doPhaseShiftSearch.get():
                args['-ExtPhase'] = f'{self.minPhaseShift} {self.maxPhaseShift}'

        tiltAxisAngle = acq.getTiltAxisAngle() or 0.0
        if ts.hasAlignment():
            # in this case we already used ts.applyTransform()
            tiltAxisAngle = 0.0

        args['-TiltAxis'] = f"{tiltAxisAngle} {self.refineTiltAxis.get() - 1}"
        args['-TiltCor'] = self.refineTiltAngles.get() - 1

        if self.sampleType.get() == LOCAL_MOTION_COORDS:
            args['-RoiFile'] = self.coordsFn
        elif self.sampleType.get() == LOCAL_MOTION_PATCHES:
            args['-Patch'] = f"{self.patchX} {self.patchY}"

        if self.roiArea.get():
            args['-Roi'] = self.roiArea.get()

        param = ' '.join([f'{k} {str(v)}' for k, v in args.items()])
        param += ' ' + self.extraParams.get()
        return param

    def readingOutput(self) -> None:
        if self.skipAlign.get():
            self.__readingOutPutTomos()
        else:
            self.__readingOutPutTsSet()


    def __readingOutPutTomos(self) -> None:
        outTomoSet = getattr(self, OUT_TOMO, None)
        if outTomoSet:
            for ts in outTomoSet:
                self.TS_read.append(ts.getTsId())
            self.info(cyanStr(f'TsIds processed: {self.TS_read}'))
        else:
            self.info(cyanStr('No tilt-series have been processed yet'))

    def __readingOutPutTsSet(self) -> None:
        outTsSet = getattr(self, OUT_TS, None)
        if outTsSet:
            for ts in outTsSet:
                self.TS_read.append(ts.getTsId())
            self.info(cyanStr(f'TsIds processed: {self.TS_read}'))
        else:
            self.info(cyanStr('No tilt-series have been processed yet'))


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
        # fileName, fileExtension = os.path.splitext(os.path.basename(tsFn))
        # baseName = removeBaseExt(tsFn).replace(EVEN, '').replace(ODD, '')
        fileExtension = ext if ext else getExt(tsFn)
        if suffix:
            suffix = suffix if suffix.startswith('_') else '_' + suffix
        return os.path.join(prefix, tsId + suffix + fileExtension)

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
                                 outputName: Literal[OUT_TS, OUT_TS_ALN] = OUT_TS) -> SetOfTiltSeries:
        outputSetOfTiltSeries = getattr(self, outputName, None)
        if outputSetOfTiltSeries:
            outputSetOfTiltSeries.enableAppend()
        else:
            suffix = "_interpolated" if outputName == OUT_TS_ALN else ""
            outputSetOfTiltSeries = self._createSetOfTiltSeries(suffix=suffix)
            outputSetOfTiltSeries.copyInfo(self._getSetOfTiltSeries())
            acq = outputSetOfTiltSeries.getAcquisition()
            if outputName == OUT_TS_ALN:
                pixSize = self._getOutputSampling()
                alignment = ALIGN_NONE
                if self.doDW.get():
                    acq.setAccumDose(0.)
                    acq.setDosePerFrame(0.)
                    acq.setTiltAxisAngle(0.)
                    outputSetOfTiltSeries.setAcquisition(acq)
            else:
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

    def _getSetOfTiltSeries(self, isPointer: bool=False) -> Union[Pointer, SetOfTiltSeries]:
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

    def _saveInterpolated(self) -> bool:
        return self.saveStack

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
