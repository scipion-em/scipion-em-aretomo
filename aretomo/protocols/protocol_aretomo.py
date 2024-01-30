# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk) [1]
# *              Federico P. de Isidro Gomez (fp.deisidro@cnb.csic.es) [2]
# *              Alberto Garcia Mena (alberto.garcia@cnb.csic.es) [2]
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
from glob import glob
import numpy as np
import time
from typing import List, Literal, Tuple, Union, Optional

from pyworkflow.protocol import params, STEPS_PARALLEL
from pyworkflow.constants import PROD
from pyworkflow.object import Set
from pyworkflow.protocol import ProtStreamingBase
import pyworkflow.utils as pwutils
from pwem.protocols import EMProtocol
from pwem.objects import Transform
from pwem.emlib.image import ImageHandler
from tomo.protocols import ProtTomoBase
from tomo.objects import (Tomogram, TiltSeries, TiltImage,
                          SetOfTomograms, SetOfTiltSeries)

from .. import Plugin
from ..convert import getTransformationMatrix, readAlnFile
from ..constants import RECON_SART, LOCAL_MOTION_COORDS, LOCAL_MOTION_PATCHES


OUT_TS = "outputSetOfTiltSeries"
OUT_TS_ALN = "outputInterpolatedSetOfTiltSeries"
OUT_TOMO = "outputSetOfTomograms"


class ProtAreTomoAlignRecon(EMProtocol, ProtTomoBase, ProtStreamingBase):
    """ Protocol for fiducial-free alignment and reconstruction for tomography available in streaming. """
    _label = 'tilt-series align and reconstruct'
    _devStatus = PROD
    _possibleOutputs = {OUT_TS: SetOfTiltSeries,
                        OUT_TS_ALN: SetOfTiltSeries,
                        OUT_TOMO: SetOfTomograms}

    def __init__(self, **args):
        EMProtocol.__init__(self, **args)
        self.stepsExecutionMode = STEPS_PARALLEL
        self.TS_read = []
        self.outputSOTSList_objID = []

        # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        # Keep this for compatibility with older plugin versions
        form.addHidden('tiltAxisAngle', params.FloatParam,
                       default=0., label='Tilt axis angle',
                       help='Note that the orientation of tilt axis is '
                            'relative to the y-axis (vertical axis of '
                            'tilt image) and rotates counter-clockwise.\n'
                            'NOTE: this is the same convention as IMOD.')
        form.addSection(label='Input')
        form.addParam('inputSetOfTiltSeries',
                      params.PointerParam,
                      pointerClass='SetOfTiltSeries',
                      important=True,
                      label='Input set of Tilt-Series',
                      help='If you choose to skip alignment, the input '
                           'tilt-series are expected to be already aligned.')

        form.addParam('skipAlign', params.BooleanParam,
                      default=False, label='Skip alignment?',
                      help='You can skip alignment if you just want to '
                           'reconstruct a tomogram from already '
                           'aligned tilt-series.')

        form.addParam('makeTomo', params.BooleanParam,
                      default=True, label='Reconstruct tomogram?',
                      help='You can skip tomogram reconstruction, so that input '
                           'tilt-series will be only aligned.')

        form.addParam('saveStack', params.BooleanParam,
                      condition="not makeTomo and not skipAlign",
                      default=True, label="Save interpolated aligned TS?",
                      help="Choose No to discard aligned stacks.")

        form.addParam('useInputProt', params.BooleanParam, default=False,
                      condition="not skipAlign",
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Use alignment from previous AreTomo run?")
        form.addParam('inputProt', params.PointerParam, allowsNull=True,
                      pointerClass="ProtAreTomoAlignRecon",
                      condition="not skipAlign and useInputProt",
                      expertLevel=params.LEVEL_ADVANCED,
                      label="Previous AreTomo run",
                      help="Use alignment from a previous AreTomo run. "
                           "The match is made using *tsId*. This option is useful "
                           "when working with odd/even tilt-series sets. "
                           "All other input alignment parameters will be ignored.")

        form.addParam('binFactor', params.IntParam,
                      default=2, label='Binning', important=True,
                      help='Binning for aligned output tilt-series / volume.')

        form.addParam('alignZ', params.IntParam, default=800,
                      condition='not skipAlign and not useInputProt',
                      important=True,
                      label='Volume height for alignment (voxels)',
                      help='Specifies Z height (*unbinned*) of the temporary volume '
                           'reconstructed for projection matching as part '
                           'of the alignment process. This value plays '
                           'an important role in alignment accuracy. This '
                           'Z height should be always smaller than tomogram '
                           'thickness and should be close to the sample '
                           'thickness.')

        form.addParam('tomoThickness', params.IntParam,
                      condition='makeTomo', important=True,
                      default=1200, label='Tomogram thickness (voxels)',
                      help='Z height of the reconstructed volume in '
                           '*unbinned* voxels.')

        form.addParam('refineTiltAngles',
                      params.EnumParam, condition="not useInputProt",
                      choices=['No', 'Measure only', 'Measure and correct'],
                      display=params.EnumParam.DISPLAY_COMBO,
                      label="Refine tilt angles?", default=1,
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
                      condition="not useInputProt",
                      choices=['No',
                               'Refine and use the refined value for the entire tilt series',
                               'Refine and calculate tilt axis at each tilt angle'],
                      display=params.EnumParam.DISPLAY_COMBO,
                      label="Refine tilt axis angle?", default=1,
                      help="Tilt axis determination is a two-step processing in AreTomo. "
                           "A single tilt axis is first calculated followed by the determination "
                           "of how tilt axis varies over the entire tilt range. The initial "
                           "value lets users enter their estimate and AreTomo refines the "
                           "estimate in [-3º, 3º] range.")

        form.addParam('outImod', params.EnumParam,
                      display=params.EnumParam.DISPLAY_COMBO,
                      choices=['No', 'Relion 4', 'Warp', 'Save locally aligned TS'],
                      default=0,
                      label="Generate extra IMOD output?",
                      help="0 - No\n1 - generate IMOD files for Relion 4\n"
                           "2 - generate IMOD files for Warp\n"
                           "3 - generate global and local-aligned tilt series stack. "
                           "High frequencies are enhanced to alleviate the attenuation "
                           "due to interpolation.")

        form.addSection(label='Extra options')
        form.addParam('doDW', params.BooleanParam, default=False,
                      label="Do dose-weighting?")
        form.addParam('reconMethod', params.EnumParam,
                      choices=['SART', 'WBP'],
                      display=params.EnumParam.DISPLAY_HLIST,
                      label="Reconstruction method", default=RECON_SART,
                      help="Choose either SART or weighted back "
                           "projection (WBP).")

        line = form.addLine("SART options", condition='reconMethod==0')
        line.addParam('SARTiter', params.IntParam, default=15,
                      label='iterations')
        line.addParam('SARTproj', params.IntParam, default=5,
                      label='projections per subset')

        form.addParam('flipInt', params.BooleanParam, default=False,
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
                      condition="not useInputProt",
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
                       condition="not useInputProt",
                       choices=['Disable local correction', 'Isolated',
                                'Well distributed'],
                       display=params.EnumParam.DISPLAY_COMBO,
                       label="Sample type", default=0,
                       help="AreTomo provides two means to correct the local "
                            "motion, one for isolated sample and the other "
                            "for well distributed across the field of view.")

        group.addParam('coordsFn', params.FileParam, default='',
                       label='Coordinate file',
                       condition='not useInputProt and sampleType==%d' % LOCAL_MOTION_COORDS,
                       help="A list of x and y coordinates should be put "
                            "into a two-column text file, one column for x "
                            "and the other for y. Each pair defines a region "
                            "of interest (ROI). The origin of the coordinate "
                            "system is at the image's lower left corner.")

        line = group.addLine("Patches",
                             condition='not useInputProt and sampleType==%d' % LOCAL_MOTION_PATCHES)
        line.addParam('patchX', params.IntParam, default=5,
                      label='X')
        line.addParam('patchY', params.IntParam, default=5,
                      label='Y')

        form.addParam('darkTol', params.FloatParam, default=0.7,
                      important=True,
                      label="Dark tolerance",
                      help="Set tolerance for removing dark images. The range is "
                           "in (0, 1). The default value is 0.7. "
                           "The higher value is more restrictive.")

        form.addParam('extraParams', params.StringParam, default='',
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Additional parameters',
                      help="Extra command line parameters. See AreTomo help.")

        form.addHidden(params.GPU_LIST, params.StringParam,
                       default='0', label="Choose GPU IDs")

        form.addParallelSection(threads=3)

    # --------------------------- INSERT steps functions ----------------------
    def stepsGeneratorStep(self) -> None:
        """
        This step should be implemented by any streaming protocol.
        It should check its input and when ready conditions are met
        call the self._insertFunctionStep method.
        """
        closeSetStepDeps=[]
        self.readingOutput()
        while True:
            listTSInput = list(self._getSetOfTiltSeries().getIdSet())
            if not self._getSetOfTiltSeries().isStreamOpen() and self.TS_read == listTSInput:
                self.info('Input set closed, all items processed\n')
                self._insertFunctionStep(self._closeOutputSet, prerquisites=closeSetStepDeps)
                break
            for ts in self._getSetOfTiltSeries():
                if ts.getObjId() not in self.TS_read:
                    self.info(f"TS_ID input: {listTSInput}\n"
                              f"TS_ID reading... {ts.getObjId()}\n"
                              f"TS_ID read: {self.TS_read}\n")
                    self.TS_read.append(ts.getObjId())
                    try:
                        args = (ts.getObjId(), ts.getTsId(),
                                ts.getFirstItem().getFileName())
                        convertInput = self._insertFunctionStep(
                            self.convertInputStep, *args, prerequisites=[])
                        runAreTomo = self._insertFunctionStep(
                            self.runAreTomoStep, *args,   prerequisites=[convertInput])
                        createOutputS = self._insertFunctionStep(
                            self.createOutputStep, *args, prerequisites=[runAreTomo])
                        closeSetStepDeps.append(createOutputS)

                    except Exception as e:
                        self.error(f'Error reading TS info: {e}')
                        self.error(f'ts.getFirstItem(): {ts.getFirstItem()}')
                time.sleep(10)

    # --------------------------- STEPS functions -----------------------------
    def convertInputStep(self, tsObjId: int, tsId: str, tsFn: str):
        self.info(f'------- convertInputStep ts_id: {tsObjId}')

        ts = self._getSetOfTiltSeries()[tsObjId]

        extraPrefix = self._getExtraPath(tsId)
        tmpPrefix = self._getTmpPath(tsId)
        pwutils.makePath(tmpPrefix)
        pwutils.makePath(extraPrefix)

        # Apply the transformation for the input tilt-series
        outputTsFileName = self.getFilePath(tsFn, tmpPrefix, ".mrc")
        rotationAngle = ts.getAcquisition().getTiltAxisAngle()
        doSwap = 45 < abs(rotationAngle) < 135
        ts.applyTransform(outputTsFileName, swapXY=doSwap)

        # Generate angle file
        angleFilePath = self.getFilePath(tsFn, tmpPrefix, ".tlt")
        self._generateTltFile(ts, angleFilePath)

        if self.useInputProt:
            # Find and copy aln file
            protExtra = self.inputProt.get()._getExtraPath(tsId)
            protAlnBase = self.getFilePath(tsFn, protExtra, ".aln").replace(
                "_even", "*").replace("_odd", "*")
            protAln = glob(protAlnBase)
            if protAln:
                pwutils.copyFile(protAln[0],
                                 self.getFilePath(tsFn, extraPrefix, ".aln"))
                self.info(f"Using input alignment: {protAln[0]}")
            else:
                raise FileNotFoundError("Missing input aln file ", protAlnBase)

    def runAreTomoStep(self, tsObjId: int, tsId: str, tsFn: str):
        """ Call AreTomo with the appropriate parameters. """
        self.info(f'------- runAreTomoStep ts_id: {tsObjId}')

        tsSet = self._getSetOfTiltSeries()
        ts = tsSet[tsObjId]

        extraPrefix = self._getExtraPath(tsId)
        tmpPrefix = self._getTmpPath(tsId)

        args = {
            '-InMrc': self.getFilePath(tsFn, tmpPrefix, ".mrc"),
            '-OutMrc': self.getFilePath(tsFn, extraPrefix, ".mrc"),
            '-OutImod': self.outImod.get(),
            '-AngFile': self.getFilePath(tsFn, tmpPrefix, ".tlt"),
            '-VolZ': self.tomoThickness if self.makeTomo else 0,
            '-OutBin': self.binFactor,
            '-FlipInt': 1 if self.flipInt else 0,
            '-FlipVol': 1 if self.makeTomo and self.flipVol else 0,
            '-PixSize': tsSet.getSamplingRate(),
            '-Kv': tsSet.getAcquisition().getVoltage(),
            '-Cs': tsSet.getAcquisition().getSphericalAberration(),
            '-Defoc': 0,  # disable defocus correction
            '-DarkTol': self.darkTol.get(),
            '-Gpu': '%(GPU)s'
        }

        if not self.useInputProt:
            args['-Align'] = 0 if self.skipAlign else 1

            tiltAxisAngle = ts.getAcquisition().getTiltAxisAngle() or 0.0
            if ts.hasAlignment():
                # in this case we already used ts.applyTransform()
                tiltAxisAngle = 0.0

            args['-TiltAxis'] = f"{tiltAxisAngle} {self.refineTiltAxis.get() - 1}"
            args['-TiltCor'] = self.refineTiltAngles.get() - 1

            if not self.skipAlign:
                args['-AlignZ'] = self.alignZ

            if self.sampleType.get() == LOCAL_MOTION_COORDS:
                args['-RoiFile'] = self.coordsFn
            elif self.sampleType.get() == LOCAL_MOTION_PATCHES:
                args['-Patch'] = f"{self.patchX} {self.patchY}"

            if self.roiArea.get():
                args['-Roi'] = self.roiArea.get()

        else:
            args['-AlnFile'] = self.getFilePath(tsFn, extraPrefix, ".aln")

        if self.reconMethod == RECON_SART:
            args['-Sart'] = f"{self.SARTiter} {self.SARTproj}"
        else:
            args['-Wbp'] = 1

        param = ' '.join([f'{k} {str(v)}' for k, v in args.items()])
        param += ' ' + self.extraParams.get()
        program = Plugin.getProgram()

        self.runJob(program, param, env=Plugin.getEnviron())

    def createOutputStep(self, tsObjId: int, tsId: str, tsFn: str):
        self.info(f'------- createOutputStep ts_id: {tsObjId}')

        ts = self._getSetOfTiltSeries()[tsObjId]
        extraPrefix = self._getExtraPath(tsId)

        if not (self.makeTomo and self.skipAlign):
            AretomoAln = readAlnFile(self.getFilePath(tsFn, extraPrefix, ".aln"))
            alignmentMatrix = getTransformationMatrix(AretomoAln.imod_matrix)

        if self.makeTomo:
            outputSetOfTomograms = self.getOutputSetOfTomograms()

            # Tomogram attributes
            newTomogram = Tomogram()
            newTomogram.setLocation(self.getFilePath(tsFn, extraPrefix, ".mrc"))
            newTomogram.setSamplingRate(outputSetOfTomograms.getSamplingRate())
            newTomogram.setOrigin()
            newTomogram.setAcquisition(ts.getAcquisition())
            newTomogram.setTsId(tsId)

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
                accumDose = 0.

                for secNum, tiltImage in enumerate(ts.iterItems(orderBy="_index")):
                    if secNum in AretomoAln.sections:
                        newTi = TiltImage()
                        newTi.copyInfo(tiltImage, copyId=False, copyTM=False)

                        acq = tiltImage.getAcquisition()
                        acq.setTiltAxisAngle(0.)
                        newTi.setAcquisition(acq)

                        secIndex = AretomoAln.sections.index(secNum)
                        newTi.setTiltAngle(AretomoAln.tilt_angles[secIndex])
                        newTi.setLocation(secIndex + 1,
                                          (self.getFilePath(tsFn, extraPrefix, ".mrc")))
                        newTi.setSamplingRate(self._getOutputSampling())
                        newTs.append(newTi)
                        accumDose = acq.getAccumDose()

                acq = newTs.getAcquisition()
                acq.setAccumDose(accumDose)  # set accum dose from the last tilt-image
                acq.setTiltAxisAngle(0.)  # 0 because TS is aligned
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
                pwutils.cleanPath(self.getFilePath(tsFn, extraPrefix, ".mrc"))

        # Save original TS stack with new alignment,
        # unless making a tomo from pre-aligned TS
        if not (self.makeTomo and self.skipAlign):
            outputSetOfTiltSeries = self.getOutputSetOfTiltSeries(OUT_TS)
            newTs = ts.clone()
            newTs.copyInfo(ts)
            newTs.setSamplingRate(self._getInputSampling())
            outputSetOfTiltSeries.append(newTs)

            for secNum, tiltImage in enumerate(ts.iterItems(orderBy="_index")):
                newTi = tiltImage.clone()
                newTi.copyInfo(tiltImage, copyId=True, copyTM=False)
                transform = Transform()

                if secNum not in AretomoAln.sections:
                    newTi.setEnabled(False)
                    transform.setMatrix(np.identity(3))
                else:
                    # set tilt angles
                    secIndex = AretomoAln.sections.index(secNum)
                    acq = tiltImage.getAcquisition()
                    acq.setTiltAxisAngle(AretomoAln.tilt_axes[secIndex])
                    newTi.setAcquisition(acq)
                    newTi.setTiltAngle(AretomoAln.tilt_angles[secIndex])

                    # set Transform
                    m = alignmentMatrix[:, :, secIndex]
                    self.debug(
                        f"Section {secNum}: {AretomoAln.tilt_axes[secIndex]}, "
                        f"{AretomoAln.tilt_angles[secIndex]}")
                    transform.setMatrix(m)

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

    # --------------------------- INFO functions ------------------------------
    def _summary(self) -> List[str]:
        summary = []
        if hasattr(self, OUT_TOMO):
            summary.append(f"Input tilt-series: "
                           f"{self._getSetOfTiltSeries().getSize()}.\n"
                           f"Tomograms reconstructed: "
                           f"{self.outputSetOfTomograms.getSize()}.\n")
        elif hasattr(self, OUT_TS):
            summary.append(f"Input tilt-series: "
                           f"{self._getSetOfTiltSeries().getSize()}.\n"
                           f"Tilt series aligned: "
                           f"{self.outputSetOfTiltSeries.getSize()}")
        else:
            summary.append("Output is not ready yet.")

        if self._saveInterpolated() and not self.makeTomo:
            summary.append("*Interpolated TS stack may have a few "
                           "dark tilt images removed.*")

        return summary

    def _methods(self) -> List[str]:
        methods = []

        return methods

    def _validate(self) -> List[str]:
        errors = []
        self._validateThreads(errors)

        if self.useInputProt:
            if not self.inputProt.hasValue():
                errors.append("Provide input AreTomo protocol for alignment.")
        else:
            if (not self.skipAlign) and self.makeTomo and (
                    self.alignZ >= self.tomoThickness):
                errors.append("Z volume height for alignment should be always "
                              "smaller than tomogram thickness.")

        if self.skipAlign and not self.makeTomo:
            errors.append("You cannot switch off both alignment and "
                          "reconstruction.")

        if self._getSetOfTiltSeries():
            if self._getSetOfTiltSeries().hasAlignment() and not self.skipAlign:
                errors.append("Input tilt-series already have alignment "
                              "information. You probably want to skip alignment step.")

        return errors

    # --------------------------- UTILS functions -----------------------------
    def readingOutput(self) -> None:
        try:
            if self.outputSetOfTiltSeries:
                for ts in self.outputSetOfTiltSeries:
                    self.TS_read.append(ts.getObjId())
            self.info(f'Tomograms calculated for this TS_ID : {self.TS_read}')
            self.outputSOTSList_objID = self.TS_read

        except AttributeError:  # There is no outputSetOfTiltSeries
            pass

    def getFilePath(self,
                    tsFn: Union[str, os.PathLike],
                    prefix: str,
                    ext: Optional[str] = None) -> Union[str, os.PathLike]:
        fileName, fileExtension = os.path.splitext(os.path.basename(tsFn))
        if ext is not None:
            fileExtension = ext

        return os.path.join(prefix, fileName + fileExtension)

    def getOutputSetOfTomograms(self) -> SetOfTomograms:
        if hasattr(self, OUT_TOMO):
            self.outputSetOfTomograms.enableAppend()
        else:
            outputSetOfTomograms = self._createSetOfTomograms()
            outputSetOfTomograms.copyInfo(self._getSetOfTiltSeries())
            outputSetOfTomograms.setSamplingRate(self._getOutputSampling())
            outputSetOfTomograms.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(**{OUT_TOMO: outputSetOfTomograms})
            self._defineSourceRelation(self.inputSetOfTiltSeries,
                                       outputSetOfTomograms)
        return self.outputSetOfTomograms

    def getOutputSetOfTiltSeries(self,
                                 outputName: Literal[OUT_TS, OUT_TS_ALN] = OUT_TS) -> SetOfTiltSeries:
        if hasattr(self, outputName):
            getattr(self, outputName).enableAppend()
        else:
            if outputName == OUT_TS_ALN:
                suffix = "_interpolated"
                pixSize = self._getOutputSampling()
            else:
                suffix = ""
                pixSize = self._getInputSampling()

            outputSetOfTiltSeries = self._createSetOfTiltSeries(suffix=suffix)
            outputSetOfTiltSeries.copyInfo(self._getSetOfTiltSeries())
            # Dimensions will be updated later
            outputSetOfTiltSeries.setDim(self._getSetOfTiltSeries().getDim())
            outputSetOfTiltSeries.setSamplingRate(pixSize)
            outputSetOfTiltSeries.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(**{outputName: outputSetOfTiltSeries})
            self._defineSourceRelation(self.inputSetOfTiltSeries,
                                       outputSetOfTiltSeries)
        return getattr(self, outputName)

    def _getSetOfTiltSeries(self) -> SetOfTiltSeries:
        return self.inputSetOfTiltSeries.get()

    def _getOutputSampling(self) -> float:
        return self._getInputSampling() * self.binFactor.get()

    def _getInputSampling(self) -> float:
        return self._getSetOfTiltSeries().getSamplingRate()

    def _getOutputDim(self, fn: str) -> Tuple[int, int, int]:
        ih = ImageHandler()
        x, y, z, _ = ih.getDimensions(fn)
        return (x, y, z)

    def _generateTltFile(self, ts: TiltSeries,
                         outputFn: os.PathLike) -> None:
        """ Generate .tlt file with tilt angles and accumulated dose. """
        angleList = []

        for ti in ts.iterItems(orderBy="_index"):
            accDose = ti.getAcquisition().getAccumDose()
            tAngle = ti.getTiltAngle()
            angleList.append((tAngle, accDose))

        with open(outputFn, 'w') as f:
            if self.doDW:
                f.writelines(f"{i[0]:0.3f} {i[1]:0.3f}\n" for i in angleList)
            else:
                f.writelines(f"{i[0]:0.3f}\n" for i in angleList)

    def _saveInterpolated(self) -> bool:
        return self.saveStack
