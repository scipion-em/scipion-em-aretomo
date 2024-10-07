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
from tomo.protocols import ProtImportTs

from tomo.tests.test_base_centralized_layer import TestBaseCentralizedLayer
from . import DataSetEmpiar10453, EMDB_10453, TS_079, TS_145

from ..protocols.protocol_aretomo import (ProtAreTomoAlignRecon, OUT_TS, OUT_TOMO,
                                          OUT_TS_ALN, OUT_CTFS)


class TestAreTomo2Base(TestBaseCentralizedLayer):
    ds = None
    importedTs = None
    unbinnedSRate = DataSetEmpiar10453.unbinnedPixSize.value
    nAngles = DataSetEmpiar10453.nAngles.value
    tsAcqDict = DataSetEmpiar10453.testTsAcqDict.value
    tsRefTAxAcqDict = DataSetEmpiar10453.testTsRefTAxAcqDict.value
    tsInterpAcqDict = DataSetEmpiar10453.testTsInterpAcqDict.value
    nTiltSeries = len(tsAcqDict)

    alignZ = 800
    binFactor = 4
    unbinnedThk = 1200

    # expectedTomoDims = [958, 926, 300]

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet(EMDB_10453)
        cls._runPreviousProtocols()
        # cls.expectedOriginShifts = list(np.array(cls.expectedTomoDims) / -2 * cls.unbinnedSRate * cls.binFactor)
        # cls.inTsSet = cls._runImportTs()

        # Acquisition of the interpolated TS: aretomo allows to dose-weight the TS, so if the interpolated TS is
        # generated and the dose-weight option is active, then the dose is set to 0 to avoid double dose correction
        # if using the interpolated TS for the PPPT. Also, the tilt axis angle should be 0 as the tilt series is
        # aligned
        # Acquisition of TS_03 interpolated with DW
        # testAcq03InterpDw = DataSetRe4STATuto.testAcq03Interp.value.clone()
        # testAcq03InterpDw.setAccumDose(0)
        # Acquisition of TS_54 interpolated with DW
        # testAcq54InterpDw = DataSetRe4STATuto.testAcq54Interp.value.clone()
        # testAcq54InterpDw.setAccumDose(0)
        # Tilt series interpolated acq dict wih DW
        # cls.tsAcqInterpDwDict = {
        #     TS_03: testAcq03InterpDw,
        #     TS_54: testAcq54InterpDw
        # }

    @classmethod
    def _runPreviousProtocols(cls):
        cls.importedTs = cls._runImportTs()

    @classmethod
    def _runImportTs(cls):
        print(magentaStr("\n==> Importing the tilt series:"))
        protTsImport = cls.newProtocol(ProtImportTs,
                                       filesPath=cls.ds.getPath(),
                                       filesPattern=DataSetEmpiar10453.tsFilesPattern.value)

        cls.launchProtocol(protTsImport)
        tsImported = getattr(protTsImport, protTsImport.OUTPUT_NAME, None)
        return tsImported


class TestAreTomo2(TestAreTomo2Base):

    def test_align_01(self):
        print(magentaStr("\n==> Testing AreTomo:"
                         "\n\t- Align only"
                         "\n\t- Generate also the interpolated TS"
                         "\n\t- Dose weighting"
                         "\n\t- No views are excluded"
                         "\n\t- CTF not generated"))

        # Expected values
        # Dose weighting
        testAcq079 = DataSetEmpiar10453.testAcq079RefTAx.value.clone()
        testAcq145 = DataSetEmpiar10453.testAcq145RefTAx.value.clone()
        testAcq079.setAccumDose = 0.
        testAcq145.setAccumDose = 0.
        testAcqDict = {TS_079: testAcq079,
                       TS_145: testAcq145}

        expectedDimsTs = DataSetEmpiar10453.getTestTsDims(nImgs=self.nAngles)
        expectedDimsInterpTs = DataSetEmpiar10453.getTestInterpTsDims(binningFactor=self.binFactor,
                                                                     nImgs=self.nAngles)

        # Run the protocol
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=self.importedTs,
                                makeTomo=False,
                                doDW=True,
                                doEstimateCtf=False,
                                alignZ=self.alignZ,
                                binFactor=self.binFactor,
                                darkTol=0.1)
        prot.setObjLabel('Align only, no CTF')
        self.launchProtocol(prot)

        # CHECK THE OUTPUTS
        # Tilt series
        self.checkTiltSeries(getattr(prot, OUT_TS, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate,
                             expectedDimensions=expectedDimsTs,
                             testAcqObj=testAcqDict,
                             hasAlignment=True,
                             alignment=ALIGN_2D,
                             anglesCount=self.nAngles)
        # Interpolated TS
        self.checkTiltSeries(getattr(prot, OUT_TS_ALN, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate * self.binFactor,
                             expectedDimensions=expectedDimsInterpTs,
                             testAcqObj=self.tsInterpAcqDict,
                             anglesCount=self.nAngles,
                             isInterpolated=True)
        # CTFs
        self.assertIsNone(getattr(prot, OUT_CTFS, None))
    #
    # def test_align_02(self):
    #     print(magentaStr("\n==> Testing AreTomo:"
    #                      "\n\t- Align only"
    #                      "\n\t- Generate also the interpolated TS"
    #                      "\n\t- No dose weighting"
    #                      "\n\t- Some views are excluded"
    #                      "\n\t- CTF generated"))
    #
    #     # Expected values
    #     exludedViews = {TS_03: [0, 1, 2, 34, 35, 36, 37, 38, 39],
    #                     TS_54: [0, 1, 2, 39, 40]}
    #     expectedDimsTsInterp = {TS_03: [958, 926, 31],
    #                             TS_54: [958, 926, 36]}
    #     nAnglesDict = {TS_03: 31,
    #                    TS_54: 36}
    #     # Update the corresponding acquisition dictionary with the refined tilt axis angle values
    #     # Note: they're different from in the previous test because of the views exclusion
    #     tsAcqDict = self.tsAcqDict
    #     tsAcqDict[TS_03].setTiltAxisAngle(84.9)
    #     tsAcqDict[TS_54].setTiltAxisAngle(85.19)
    #     # Because some of the excluded views are at the end of the stack and DW is not applied in this test, the TS
    #     # accumulated values will be updated in the interpolated TS
    #     tsAcqDictInterp = DataSetRe4STATuto.tsAcqInterpDict.value
    #     tsAcqDictInterp[TS_03].setAccumDose(96)
    #     tsAcqDictInterp[TS_54].setAccumDose(111)
    #
    #     # Run the protocol
    #     prot = self.newProtocol(ProtAreTomoAlignRecon,
    #                             inputSetOfTiltSeries=self.inTsSet,
    #                             makeTomo=False,
    #                             alignZ=self.alignZ,
    #                             binFactor=self.binFactor,
    #                             darkTol=0.9)
    #     prot.setObjLabel('Align only, exclude views')
    #     self.launchProtocol(prot)
    #
    #     # CHECK THE OUTPUTS
    #     # Tilt series
    #     self.checkTiltSeries(getattr(prot, OUT_TS, None),
    #                          expectedSetSize=self.nTiltSeries,
    #                          expectedSRate=self.unbinnedSRate,
    #                          expectedDimensions=self.expectedDimsTs,
    #                          testAcqObj=tsAcqDict,
    #                          hasAlignment=True,
    #                          alignment=ALIGN_2D,
    #                          anglesCount=self.nAnglesDict,
    #                          excludedViewsDict=exludedViews)
    #
    #     # Interpolated TS
    #     self.checkTiltSeries(getattr(prot, OUT_TS_ALN, None),
    #                          expectedSetSize=self.nTiltSeries,
    #                          expectedSRate=self.unbinnedSRate * self.binFactor,
    #                          # Protocol sets the bin factor to 2 by default
    #                          expectedDimensions=expectedDimsTsInterp,
    #                          testAcqObj=tsAcqDictInterp,
    #                          anglesCount=nAnglesDict,
    #                          isInterpolated=True)
    #     # CTFs
    #     self._checkCTFs(getattr(prot, OUT_CTFS, None), excludedViewsDict=exludedViews)
    #
    # def test_alignAndReconstruct(self):
    #     print(magentaStr("\n==> Testing AreTomo:"
    #                      "\n\t- Align and reconstruct"
    #                      "\n\t- Interpolated TS not generated"
    #                      "\n\t- No dose weighting"
    #                      "\n\t- No views are excluded"
    #                      "\n\t- CTF generated"))
    #
    #     # Update the corresponding acquisition dictionary with the refined tilt axis angle values
    #     tsAcqDict = self.tsAcqDict
    #     tsAcqDict[TS_03].setTiltAxisAngle(85.13)
    #     tsAcqDict[TS_54].setTiltAxisAngle(85.3)
    #
    #     # Run the protocol
    #     prot = self.newProtocol(ProtAreTomoAlignRecon,
    #                             inputSetOfTiltSeries=self.inTsSet,
    #                             tomoThickness=self.unbinnedThk,
    #                             alignZ=self.alignZ,
    #                             binFactor=self.binFactor,
    #                             darkTol=0.1)
    #     prot.setObjLabel('Align & rec')
    #     self.launchProtocol(prot)
    #
    #     # CHECK THE OUTPUTS
    #     # Tilt series
    #     self.checkTiltSeries(getattr(prot, OUT_TS, None),
    #                          expectedSetSize=self.nTiltSeries,
    #                          expectedSRate=self.unbinnedSRate,
    #                          expectedDimensions=self.expectedDimsTs,
    #                          testAcqObj=tsAcqDict,
    #                          hasAlignment=True,
    #                          alignment=ALIGN_2D,
    #                          anglesCount=self.nAnglesDict)
    #
    #     # CTFs
    #     self._checkCTFs(getattr(prot, OUT_CTFS, None))
    #
    #     # Tomograms
    #     self.checkTomograms(getattr(prot, OUT_TOMO, None),
    #                         expectedSetSize=self.nTiltSeries,
    #                         expectedSRate=self.unbinnedSRate * self.binFactor,
    #                         expectedDimensions=self.expectedTomoDims,
    #                         expectedOriginShifts=self.expectedOriginShifts)
    #
    # def _checkCTFs(self, ctfSet, excludedViewsDict=None):
    #     self.checkCTFs(ctfSet,
    #                    expectedSetSize=self.nTiltSeries,
    #                    excludedViewsDict=excludedViewsDict,
    #                    expectedPsdFile=True)
