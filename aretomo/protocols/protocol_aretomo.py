# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk) [1]
# *              Federico P. de Isidro Gomez (fp.deisidro@cnb.csic.es) [2]
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

import pyworkflow.protocol.params as params
from pyworkflow.constants import BETA
from pyworkflow.object import Set
import pyworkflow.utils as pwutils
from pwem.protocols import EMProtocol
from pwem.objects import Transform
from pwem.emlib.image import ImageHandler

from tomo.protocols import ProtTomoBase
from tomo.objects import Tomogram, TomoAcquisition, TiltSeries, TiltImage

from .. import Plugin
from ..constants import *
from ..convert import getTransformationMatrix


class ProtAreTomoAlignRecon(EMProtocol, ProtTomoBase):
    """ Protocol for fiducial-free alignment and reconstruction for tomography.

    Find more information at https://msg.ucsf.edu/software
    """
    _label = 'tilt-series align and reconstruct'
    _devStatus = BETA

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
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

        form.addParam('binFactor', params.IntParam,
                      default=1, label='Binning', important=True,
                      help='Binning for aligned output tilt-series / volume.')

        form.addParam('alignZ', params.IntParam, default=800,
                      condition='not skipAlign', important=True,
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
                      default=1000, label='Tomogram thickness (voxels)',
                      help='Z height of the reconstructed volume in '
                           '*unbinned* voxels.')

        form.addParam('tiltAxisAngle', params.FloatParam,
                      default=0., label='Tilt axis angle',
                      help='Note that the orientation of tilt axis is '
                           'relative to the y-axis (vertical axis of '
                           'tilt image) and rotates counter-clockwise.\n'
                           'NOTE: this is the same convention as IMOD.')

        form.addParam('doDW', params.BooleanParam, default=False,
                      label="Do dose-weighting?")

        form.addSection(label='Extra options')
        form.addParam('reconMethod', params.EnumParam,
                      choices=['SART', 'WBP'],
                      display=params.EnumParam.DISPLAY_HLIST,
                      label="Reconstruction method", default=RECON_SART,
                      help="Choose either SART or weighted back "
                           "projection (WBP).")

        line = form.addLine("SART options", condition='reconMethod==0')
        line.addParam('SARTiter', params.IntParam, default=20,
                      label='iterations')
        line.addParam('SARTproj', params.IntParam, default=5,
                      label='projections per subset')

        form.addParam('flipInt', params.BooleanParam, default=False,
                      label="Flip intensity?",
                      help="By default, the reconstructed volume "
                           "and the input tilt series use the same grayscale "
                           "that makes dense structures dark.")

        form.addParam('flipVol', params.BooleanParam, default=True,
                      label="Flip volume?",
                      help="This saves x-y volume slices according to their Z "
                           "coordinates, similar to IMOD.")

        form.addParam('roiArea', params.StringParam, default='',
                      label="ROI for focused alignment",
                      help="By default AreTomo assumes the region of interest "
                           "at the center of 0° projection image. A circular "
                           "mask is employed to down-weight the area outside "
                           "ROI during the alignment. When the structures of "
                           "interest are far away from the tilt axis, the "
                           "angular error in the determination of tilt axis "
                           "will significantly amplify the translational error. "
                           "ROI function can effectively improve the alignment "
                           "accuracy for the distant structures.\nHere you can "
                           "provide *a pair of x and y coordinates*, representing "
                           "the center of the region of interest.\n"
                           "The region of interest should be selected from 0° "
                           "projection image with the origin at the lower left "
                           "corner. IMOD's Pixel View is a good tool to select "
                           "the center of region of interest.")

        group = form.addGroup('Local motion correction')
        group.addParam('sampleType', params.EnumParam,
                       choices=['Disable local correction', 'Isolated', 'Well distributed'],
                       display=params.EnumParam.DISPLAY_COMBO,
                       label="Sample type", default=0,
                       help="AreTomo provides two means to correct the local "
                            "motion, one for isolated sample and the other "
                            "for well distributed across the field of view.")

        group.addParam('coordsFn', params.FileParam, default='',
                       label='Coordinate file',
                       condition='sampleType==%d' % LOCAL_MOTION_COORDS,
                       help="A list of x and y coordinates should be put "
                            "into a two-column text file, one column for x "
                            "and the other for y. Each pair defines a region "
                            "of interest (ROI). The origin of the coordinate "
                            "system is at the image’s lower left corner.")

        line = group.addLine("Patches",
                             condition='sampleType==%d' % LOCAL_MOTION_PATCHES)
        line.addParam('patchX', params.IntParam, default=5,
                      label='X')
        line.addParam('patchY', params.IntParam, default=5,
                      label='Y')

        form.addParam('extraParams', params.StringParam, default='',
                      expertLevel=params.LEVEL_ADVANCED,
                      label='Additional parameters',
                      help="Extra command line parameters. See AreTomo help.")

        form.addHidden(params.GPU_LIST, params.StringParam,
                       default='0', label="Choose GPU IDs")

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        for ts in self.inputSetOfTiltSeries.get():
            self._insertFunctionStep('convertInputStep', ts.getObjId())
            self._insertFunctionStep('runAreTomoStep', ts.getObjId())
            self._insertFunctionStep('createOutputStep', ts.getObjId())
        self._insertFunctionStep('closeOutputSetsStep')

    # --------------------------- STEPS functions -----------------------------
    def convertInputStep(self, tsObjId):
        ts = self._getSetOfTiltSeries()[tsObjId]
        tsId = ts.getTsId()
        extraPrefix = self._getExtraPath(tsId)
        tmpPrefix = self._getTmpPath(tsId)
        pwutils.makePath(tmpPrefix)
        pwutils.makePath(extraPrefix)

        """Apply the transformation for the input tilt-series"""
        outputTsFileName = self.getFilePath(tsObjId, tmpPrefix, ".mrc")
        ts.applyTransform(outputTsFileName)

        """Generate angle file"""
        angleFilePath = self.getFilePath(tsObjId, tmpPrefix, ".tlt")
        self._generateTltFile(ts, angleFilePath)

    def runAreTomoStep(self, tsObjId):
        """ Call AreTomo with the appropriate parameters. """
        tsSet = self._getSetOfTiltSeries()
        ts = tsSet[tsObjId]
        tsId = ts.getTsId()

        extraPrefix = self._getExtraPath(tsId)
        tmpPrefix = self._getTmpPath(tsId)

        args = {
            '-InMrc': self.getFilePath(tsObjId, tmpPrefix, ".mrc"),
            '-OutMrc': self.getFilePath(tsObjId, extraPrefix, ".mrc"),
            '-AngFile': self.getFilePath(tsObjId, tmpPrefix, ".tlt"),
            '-TiltAxis': self.tiltAxisAngle.get(),
            '-TiltRange': '%d %d' % (ts.getFirstItem().getTiltAngle(),
                                     ts[ts.getSize()].getTiltAngle()),
            '-VolZ': self.tomoThickness if self.makeTomo else 0,
            '-OutBin': self.binFactor,
            '-Align': 0 if self.skipAlign else 1,
            '-FlipInt': 1 if self.flipInt else 0,
            '-FlipVol': 1 if self.flipVol else 0,
            '-PixSize': tsSet.getSamplingRate(),
            '-Kv': tsSet.getAcquisition().getVoltage(),
            '-Cs': tsSet.getAcquisition().getSphericalAberration(),
            '-OutXF': 1,  # generate IMOD-compatible file
            '-Defoc': 0,  # disable defocus correction
            '-Gpu': '%(GPU)s'
        }

        if not self.skipAlign:
            args['-AlignZ'] = self.alignZ

        if self.reconMethod == RECON_SART:
            args['-Sart'] = '%d %d' % (self.SARTiter, self.SARTproj)
        else:
            args['-Wbp'] = 1

        if self.roiArea.get():
            args['-Roi'] = self.roiArea.get()

        if self.sampleType.get() == LOCAL_MOTION_COORDS:
            args['-RoiFile'] = self.coordsFn
        elif self.sampleType.get() == LOCAL_MOTION_PATCHES:
            args['-Patch'] = '%d %d' % (self.patchX, self.patchY)

        param = ' '.join(['%s %s' % (k, str(v)) for k, v in args.items()])
        param += ' ' + self.extraParams.get()
        program = Plugin.getProgram()

        self.runJob(program, param, env=Plugin.getEnviron())

    def createOutputStep(self, tsObjId):
        ts = self._getSetOfTiltSeries()[tsObjId]
        tsId = ts.getTsId()
        extraPrefix = self._getExtraPath(tsId)

        if self.makeTomo:
            outputSetOfTomograms = self.getOutputSetOfTomograms()

            newTomogram = Tomogram()
            newTomogram.setLocation(self.getFilePath(tsObjId, extraPrefix, ".mrc"))

            # Set tomogram origin
            origin = Transform()
            sr = self._getOutputSampling()
            origin.setShifts(ts.getFirstItem().getXDim() / -2. * sr,
                             ts.getFirstItem().getYDim() / -2. * sr,
                             self.tomoThickness.get() / self.binFactor.get() / -2 * sr)
            newTomogram.setOrigin(origin)

            # Set tomogram acquisition
            acquisition = TomoAcquisition()
            acquisition.setAngleMin(ts.getFirstItem().getTiltAngle())
            acquisition.setAngleMax(ts[ts.getSize()].getTiltAngle())
            acquisition.setStep(self.getAngleStepFromSeries(ts))
            newTomogram.setAcquisition(acquisition)

            outputSetOfTomograms.append(newTomogram)
            outputSetOfTomograms.update(newTomogram)
            outputSetOfTomograms.write()
        else:
            outputSetOfTiltSeries = self.getOutputSetOfTiltSeries()

            newTs = TiltSeries(tsId=tsId)
            newTs.copyInfo(ts)
            outputSetOfTiltSeries.append(newTs)
            newTs.setSamplingRate(self._getOutputSampling())

            secs = self._readAlnFile(self.getFilePath(tsObjId, extraPrefix, ".aln"))

            for index, tiltImage in enumerate(ts):
                if (index + 1) in secs:
                    newTi = TiltImage()
                    newTi.copyInfo(tiltImage)
                    newTi.setLocation(index + 1,
                                      (self.getFilePath(tsObjId, extraPrefix, ".mrc")))
                    newTi.setSamplingRate(self._getOutputSampling())

                    # set Transform
                    alignFn = self.getFilePath(tsObjId, extraPrefix, ".xf")
                    alignmentMatrix = getTransformationMatrix(alignFn)
                    transform = Transform()
                    transform.setMatrix(alignmentMatrix[:, :, secs.index(index+1)])
                    newTi.setTransform(transform)

                    newTs.append(newTi)

            dims = self._getOutputDim(newTi.getFileName())
            newTs.setDim(dims)
            newTs.write(properties=False)

            outputSetOfTiltSeries.update(newTs)
            outputSetOfTiltSeries.updateDim()
            outputSetOfTiltSeries.write()

        self._store()

    def closeOutputSetsStep(self):
        if self.makeTomo:
            self.getOutputSetOfTomograms().setStreamState(Set.STREAM_CLOSED)
        else:
            self.getOutputSetOfTiltSeries().setStreamState(Set.STREAM_CLOSED)
        self._store()

    # --------------------------- INFO functions ------------------------------
    def _summary(self):
        summary = []
        if hasattr(self, 'outputSetOfTomograms'):
            summary.append("Input Tilt-Series: %d.\nTomograms reconstructed: %d.\n"
                           % (self._getSetOfTiltSeries().getSize(),
                              self.outputSetOfTomograms.getSize()))
        elif hasattr(self, 'outputSetOfTiltSeries'):
            summary.append("Input Tilt-Series: %d.\nTilt series aligned: %d.\n"
                           % (self._getSetOfTiltSeries().getSize(),
                              self.outputSetOfTiltSeries.getSize()))
        else:
            summary.append("Output is not ready yet.")
        return summary

    def _methods(self):
        methods = []

        return methods
    
    def _validate(self):
        errors = []

        if not self.skipAlign and (self.alignZ >= self.tomoThickness):
            errors.append("Z volume height for alignment should be always "
                          "smaller than tomogram thickness.")

        if self.skipAlign and not self.makeTomo:
            errors.append("You cannot switch off both alignment and "
                          "reconstruction.")

        tr = self._getSetOfTiltSeries().getFirstItem().getFirstItem().hasTransform()
        if tr and not self.skipAlign:
            errors.append("Input tilt-series already have alignment "
                          "information. You probably want to skip alignment step.")

        return errors
    
    # --------------------------- UTILS functions -----------------------------
    def getFilePath(self, tsObjId, prefix, ext=None):
        ts = self._getSetOfTiltSeries()[tsObjId]
        return os.path.join(prefix,
                            ts.getFirstItem().parseFileName(extension=ext))

    def getOutputSetOfTomograms(self):
        if hasattr(self, "outputSetOfTomograms"):
            self.outputSetOfTomograms.enableAppend()
        else:
            outputSetOfTomograms = self._createSetOfTomograms()
            outputSetOfTomograms.copyInfo(self._getSetOfTiltSeries())
            outputSetOfTomograms.setSamplingRate(self._getOutputSampling())
            outputSetOfTomograms.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(outputSetOfTomograms=outputSetOfTomograms)
            self._defineSourceRelation(self.inputSetOfTiltSeries,
                                       outputSetOfTomograms)
        return self.outputSetOfTomograms

    def getOutputSetOfTiltSeries(self):
        if hasattr(self, "outputSetOfTiltSeries"):
            self.outputSetOfTiltSeries.enableAppend()
        else:
            outputSetOfTiltSeries = self._createSetOfTiltSeries()
            outputSetOfTiltSeries.copyInfo(self._getSetOfTiltSeries())
            # Dimensions will be updated later
            outputSetOfTiltSeries.setDim(self._getSetOfTiltSeries().getDim())
            outputSetOfTiltSeries.setSamplingRate(self._getOutputSampling())
            outputSetOfTiltSeries.setStreamState(Set.STREAM_OPEN)
            self._defineOutputs(outputSetOfTiltSeries=outputSetOfTiltSeries)
            self._defineSourceRelation(self.inputSetOfTiltSeries,
                                       outputSetOfTiltSeries)
        return self.outputSetOfTiltSeries

    def getAngleStepFromSeries(self, ts):
        """ This method return the average angles step from a series. """
        angleStepAverage = 0
        for i in range(1, ts.getSize()):
            angleStepAverage += abs(ts[i].getTiltAngle() - ts[i+1].getTiltAngle())

        angleStepAverage /= ts.getSize() - 1

        return angleStepAverage

    def _getSetOfTiltSeries(self):
        return self.inputSetOfTiltSeries.get()

    def _getOutputSampling(self):
        return self._getSetOfTiltSeries().getSamplingRate() * self.binFactor.get()

    def _getOutputDim(self, fn):
        ih = ImageHandler()
        x, y, z, _ = ih.getDimensions(fn)
        return (x, y, z)

    def _generateTltFile(self, ts, outputFn):
        """ Generate .tlt file with tilt angles and accumulated dose. """
        tsList = []

        for index, ti in enumerate(ts):
            accDose = ti.getAcquisition().getDosePerFrame()
            tAngle = ti.getTiltAngle()
            tsList.append((tAngle, accDose))

        with open(outputFn, 'w') as f:
            if self.doDW:
                for i in tsList:
                    f.write("%0.3f %0.3f\n" % (i[0], i[1]))
            else:
                f.writelines("%0.3f\n" % i[0] for i in tsList)

    def _readAlnFile(self, fn):
        """ Read output tilt section numbers as 1-based index. """
        secs = []
        with open(fn, 'r') as f:
            line = f.readline()
            while line:
                if not line.startswith("#"):
                    secs.append(int(line.strip().split()[0]) + 1)
                line = f.readline()

        return secs
