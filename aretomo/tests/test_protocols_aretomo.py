# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk)
# *
# * MRC Laboratory of Molecular Biology (MRC-LMB)
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
from pwem import ALIGN_2D
from pyworkflow.utils import magentaStr
from pyworkflow.tests import DataSet, setupTestProject
from tomo.objects import TomoAcquisition
from tomo.protocols import ProtImportTs, ProtImportTsBase
from tomo.tests.test_base_centralized_layer import TestBaseCentralizedLayer
from ..protocols import ProtAreTomoAlignRecon
from ..protocols.protocol_aretomo import OUT_TS, OUT_TOMO, OUT_TS_ALN, OUT_CTFS


class TestAreTomoBase(TestBaseCentralizedLayer):
    protImportTS = None
    inputDataSet = None
    inputSoTS = None
    nTiltSeries = 2
    nTiltImages = 61
    tomoThkBin2 = 200
    unbinnedSRate = 20.2
    unbinnedTsDims = [512, 512, 61]
    bin2TsDims = [256, 256, 61]
    bin2TomoDims = [256, 256, 200]
    bin2SRate = 40.4
    tiltAxisAngle = -12.5
    voltage = 300
    sphericalAb = 2.7
    amplitudeContrast = 0.1
    magnification = 105000
    initialDose = 0
    dosePerTiltImg = 0.3
    accumDose = 18.3

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.inputDataSet = DataSet.getDataSet('tomo-em')
        cls.inputSoTS = cls.inputDataSet.getFile('ts1')
        cls.testAcq = TomoAcquisition(voltage=cls.voltage,
                                      sphericalAberration=cls.sphericalAb,
                                      amplitudeContrast=cls.amplitudeContrast,
                                      magnification=cls.magnification,
                                      tiltAxisAngle=cls.tiltAxisAngle,
                                      doseInitial=cls.initialDose,
                                      dosePerFrame=cls.dosePerTiltImg,
                                      accumDose=cls.accumDose)

    @classmethod
    def _runImportTiltSeries(cls):
        print(magentaStr("\n==> Importing data - TiltSeries:"))
        cls.protImportTS = cls.newProtocol(ProtImportTs,
                                           filesPath=cls.inputDataSet.getFile('etomo'),
                                           filesPattern="BB{TS}.st",
                                           anglesFrom=0,
                                           voltage=cls.voltage,
                                           magnification=cls.magnification,
                                           sphericalAberration=cls.sphericalAb,
                                           amplitudeContrast=cls.amplitudeContrast,
                                           samplingRate=cls.unbinnedSRate,
                                           doseInitial=cls.initialDose,
                                           dosePerFrame=cls.dosePerTiltImg,
                                           minAngle=-55,
                                           maxAngle=65.0,
                                           stepAngle=2.0,
                                           tiltAxisAngle=cls.tiltAxisAngle)
        cls.launchProtocol(cls.protImportTS)
        return cls.protImportTS


class TestAreTomo(TestAreTomoBase):
    def test_alignAndReconstruct(self):
        protImport = self._runImportTiltSeries()
        print(magentaStr("\n==> Testing AreTomo:"
                         "\n\t- Align and reconstruct"
                         "\n\t- Interpolated TS not generated"
                         "\n\t- No views are excluded"))
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=getattr(protImport, ProtImportTsBase.OUTPUT_NAME, None),
                                tomoThickness=self.tomoThkBin2,
                                alignZ=180,
                                tiltAxisAngle=self.tiltAxisAngle,
                                darkTol=0.1)
        self.launchProtocol(prot)
        # CHECK THE OUTPUTS----------------------------------------------------------------
        # Non-interpolated TS
        self._checkNonInterpTsSet(getattr(prot, OUT_TS, None))

        # CTFs
        self._checkCTFs(getattr(prot, OUT_CTFS, None))

        # Tomograms
        self.checkTomograms(getattr(prot, OUT_TOMO, None),
                            expectedSetSize=self.nTiltSeries,
                            expectedSRate=self.bin2SRate,
                            expectedDimensions=self.bin2TomoDims)

    def test_align(self):
        protImport = self._runImportTiltSeries()
        print(magentaStr("\n==> Testing AreTomo:"
                         "\n\t- Align only"
                         "\n\t- Generate also the interpolated TS"
                         "\n\t- No views are excluded"))
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=getattr(protImport, ProtImportTsBase.OUTPUT_NAME, None),
                                makeTomo=False,
                                alignZ=180,
                                tiltAxisAngle=self.tiltAxisAngle,
                                darkTol=0.1)
        self.launchProtocol(prot)
        # CHECK THE OUTPUTS----------------------------------------------------------------
        # Non-interpolated TS
        self._checkNonInterpTsSet(getattr(prot, OUT_TS, None))

        # Interpolated TS
        self.checkTiltSeries(getattr(prot, OUT_TS_ALN, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.bin2SRate,  # Protocol sets the bin factor to 2 by default
                             expectedDimensions=self.bin2TsDims,
                             testAcqObj=self.testAcq,
                             anglesCount=self.nTiltImages,
                             isInterpolated=True)
        # CTFs
        self._checkCTFs(getattr(prot, OUT_CTFS, None))

    def _checkNonInterpTsSet(self, tsSet):
        self.checkTiltSeries(tsSet,
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate,
                             expectedDimensions=self.unbinnedTsDims,
                             testAcqObj=self.testAcq,
                             hasAlignment=True,
                             alignment=ALIGN_2D,
                             anglesCount=self.nTiltImages)

    def _checkCTFs(self, ctfSet):
        self.checkCTFs(ctfSet,
                       expectedSetSize=self.nTiltSeries)
