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
import numpy as np

from pwem import ALIGN_2D
from pyworkflow.utils import magentaStr
from pyworkflow.tests import DataSet, setupTestProject
from tomo.objects import TomoAcquisition
from tomo.protocols import ProtImportTs
from tomo.tests import RE4_STA_TUTO, DataSetRe4STATuto
from tomo.tests.test_base_centralized_layer import TestBaseCentralizedLayer
from ..protocols import ProtAreTomoAlignRecon
from ..protocols.protocol_aretomo import OUT_TS, OUT_TOMO, OUT_TS_ALN, OUT_CTFS


class TestAreTomoBase(TestBaseCentralizedLayer):
    ds = None
    importedTs = None
    particlesUnbinnedBoxSize = 256
    particlesExtractedBoxSize = 64
    alignZ = 900
    nTiltSeries = 2
    binFactor = 4
    unbinnedSRate = DataSetRe4STATuto.unbinnedPixSize.value
    unbinnedThk = 1200
    expectedDimsTs = {'TS_03': [3710, 3838, 40],
                      'TS_54': [3710, 3838, 41]}
    expectedTomoDims = [958, 926, 300]
    testAcq = TomoAcquisition(voltage=DataSetRe4STATuto.voltage.value,
                              sphericalAberration=DataSetRe4STATuto.sphericalAb.value,
                              amplitudeContrast=DataSetRe4STATuto.amplitudeContrast.value,
                              magnification=DataSetRe4STATuto.magnification.value,
                              tiltAxisAngle=DataSetRe4STATuto.tiltAxisAngle.value,
                              doseInitial=DataSetRe4STATuto.initialDose.value,
                              dosePerFrame=DataSetRe4STATuto.dosePerTiltImg.value,
                              accumDose=DataSetRe4STATuto.accumDose.value
                              )

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet(RE4_STA_TUTO)
        cls.expectedOriginShifts = list(np.array(cls.expectedTomoDims) / -2 * cls.unbinnedSRate * cls.binFactor)
        cls.inTsSet = cls._runImportTs()

    @classmethod
    def _runImportTs(cls):
        print(magentaStr("\n==> Importing the tilt series:"))
        protTsImport = cls.newProtocol(ProtImportTs,
                                       filesPath=cls.ds.getFile(DataSetRe4STATuto.tsPath.value),
                                       filesPattern=DataSetRe4STATuto.tsPattern.value,
                                       exclusionWords=DataSetRe4STATuto.exclusionWordsTs03ts54.value,
                                       anglesFrom=2,  # From tlt file
                                       voltage=DataSetRe4STATuto.voltage.value,
                                       magnification=DataSetRe4STATuto.magnification.value,
                                       sphericalAberration=DataSetRe4STATuto.sphericalAb.value,
                                       amplitudeContrast=DataSetRe4STATuto.amplitudeContrast.value,
                                       samplingRate=DataSetRe4STATuto.unbinnedPixSize.value,
                                       doseInitial=DataSetRe4STATuto.initialDose.value,
                                       dosePerFrame=DataSetRe4STATuto.dosePerTiltImg.value,
                                       tiltAxisAngle=DataSetRe4STATuto.tiltAxisAngle.value)

        cls.launchProtocol(protTsImport)
        tsImported = getattr(protTsImport, protTsImport.OUTPUT_NAME, None)
        return tsImported


class TestAreTomo(TestAreTomoBase):

    def test_align_01(self):
        expectedDimsTsInterp = {'TS_03': [958, 926, 40],
                                'TS_54': [958, 926, 41]}

        print(magentaStr("\n==> Testing AreTomo:"
                         "\n\t- Align only"
                         "\n\t- Generate also the interpolated TS"
                         "\n\t- No views are excluded"
                         "\n\t- CTF not generated"))
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=self.inTsSet,
                                makeTomo=False,
                                doEstimateCtf=False,
                                alignZ=self.alignZ,
                                binFactor=self.binFactor,
                                darkTol=0.1)
        prot.setObjLabel('Align only, no CTF')
        self.launchProtocol(prot)

        # CHECK THE OUTPUTS
        # Non-interpolated TS
        self._checkNonInterpTsSet(getattr(prot, OUT_TS, None))

        # Interpolated TS
        self.checkTiltSeries(getattr(prot, OUT_TS_ALN, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate * self.binFactor,  # Protocol sets the bin factor to 2 by default
                             expectedDimensions=expectedDimsTsInterp,
                             testAcqObj=self.testAcq,
                             isInterpolated=True)
        # CTFs
        self.assertIsNone(getattr(prot, OUT_CTFS, None))

    def test_align_02(self):
        exludedViews = {'TS_03': [0, 1, 2, 34, 35, 36, 37, 38, 39],
                        'TS_54': [0, 1, 2, 39, 40]}
        expectedDimsTsInterp = {'TS_03': [958, 926, 31],
                                'TS_54': [958, 926, 36]}

        print(magentaStr("\n==> Testing AreTomo:"
                         "\n\t- Align only"
                         "\n\t- Generate also the interpolated TS"
                         "\n\t- Some views are excluded"
                         "\n\t- CTF generated"))
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=self.inTsSet,
                                makeTomo=False,
                                alignZ=self.alignZ,
                                binFactor=self.binFactor,
                                darkTol=0.9)
        prot.setObjLabel('Align only, exclude views')
        self.launchProtocol(prot)

        # CHECK THE OUTPUTS
        # Non-interpolated TS
        self._checkNonInterpTsSet(getattr(prot, OUT_TS, None), excludedViewsDict=exludedViews)

        # Interpolated TS
        self.checkTiltSeries(getattr(prot, OUT_TS_ALN, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate * self.binFactor,
                             # Protocol sets the bin factor to 2 by default
                             expectedDimensions=expectedDimsTsInterp,
                             testAcqObj=self.testAcq,
                             isInterpolated=True)
        # CTFs
        self._checkCTFs(getattr(prot, OUT_CTFS, None), excludedViewsDict=exludedViews)

    def test_alignAndReconstruct(self):
        print(magentaStr("\n==> Testing AreTomo:"
                         "\n\t- Align and reconstruct"
                         "\n\t- Interpolated TS not generated"
                         "\n\t- No views are excluded"
                         "\n\t- CTF generated"))
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=self.inTsSet,
                                tomoThickness=self.unbinnedThk,
                                alignZ=self.alignZ,
                                binFactor=self.binFactor,
                                darkTol=0.1)
        prot.setObjLabel('Align & rec')
        self.launchProtocol(prot)
        # CHECK THE OUTPUTS----------------------------------------------------------------
        # Non-interpolated TS
        self._checkNonInterpTsSet(getattr(prot, OUT_TS, None))

        # CTFs
        self._checkCTFs(getattr(prot, OUT_CTFS, None))

        # Tomograms
        self.checkTomograms(getattr(prot, OUT_TOMO, None),
                            expectedSetSize=self.nTiltSeries,
                            expectedSRate=self.unbinnedSRate * self.binFactor,
                            expectedDimensions=self.expectedTomoDims,
                            expectedOriginShifts=self.expectedOriginShifts)

    def _checkNonInterpTsSet(self, tsSet, excludedViewsDict=None):
        self.checkTiltSeries(tsSet,
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate,
                             expectedDimensions=self.expectedDimsTs,
                             testAcqObj=self.testAcq,
                             hasAlignment=True,
                             alignment=ALIGN_2D,
                             excludedViewsDict=excludedViewsDict)

    def _checkCTFs(self, ctfSet, excludedViewsDict=None):
        self.checkCTFs(ctfSet,
                       expectedSetSize=self.nTiltSeries,
                       excludedViewsDict=excludedViewsDict)
