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
    unbinnedSRate = DataSetEmpiar10453.unbinnedPixSize.value
    nAngles = DataSetEmpiar10453.nAngles.value
    tsAcqDict = DataSetEmpiar10453.testTsAcqDict.value
    tsRefTAxAcqDict = DataSetEmpiar10453.testTsRefTAxAcqDict.value
    tsInterpAcqDict = DataSetEmpiar10453.testTsInterpAcqDict.value
    tsInterpDWAcqDict = DataSetEmpiar10453.testTsInterpDWAcqDict.value
    nTiltSeries = len(tsAcqDict)

    alignZ = 800
    binFactor = 4
    unbinnedThk = 1200

    expectedTomoDims = [1022, 1440, 300]

    excludedViewsDict = {
        TS_079: [0, 1, 38, 40],
        TS_145: [0, 1, 2, 39, 40]
    }


    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet(EMDB_10453)
        cls.expectedDimsTs = DataSetEmpiar10453.getTestTsDims(nImgs=cls.nAngles)
        cls.expectedOriginShifts = list(np.array(cls.expectedTomoDims) / -2 * cls.unbinnedSRate * cls.binFactor)

    @classmethod
    def _runImportTs(cls):
        print(magentaStr("\n==> Importing the tilt series:"))
        protTsImport = cls.newProtocol(ProtImportTs,
                                       filesPath=cls.ds.getPath(),
                                       filesPattern=DataSetEmpiar10453.tsFilesPattern.value)

        cls.launchProtocol(protTsImport)
        tsImported = getattr(protTsImport, protTsImport.OUTPUT_NAME, None)
        return tsImported

    @classmethod
    def _excludeTsSetViews(cls, tsSet):
        tsList = [ts.clone(ignoreAttrs=[]) for ts in tsSet]
        for ts in tsList:
            cls._excludeTsViews(tsSet, ts, cls.excludedViewsDict[ts.getTsId()])

    @staticmethod
    def _excludeTsViews(tsSet, ts, excludedViewsList):
        tiList = [ti.clone() for ti in ts]
        for i, ti in enumerate(tiList):
            if i in excludedViewsList:
                ti._objEnabled = False
                ts.update(ti)
        ts.write()
        tsSet.update(ts)
        tsSet.write()


class TestAreTomo2(TestAreTomo2Base):

    def test_align_01(self):
        # Expected values
        expectedDimsInterpTs = DataSetEmpiar10453.getTestInterpTsDims(binningFactor=self.binFactor,
                                                                      nImgs=self.nAngles)
        # Run the test
        importedTs = self._runImportTs()
        self._run_test_align_01(importedTs, expectedDimsInterpTs)

    def test_align_01_excluded_views(self):
        # Expected values
        nImgs_079 = self.nAngles - len(self.excludedViewsDict[TS_079])
        nImgs_145 = self.nAngles - len(self.excludedViewsDict[TS_145])
        expectedDimsInterpTs = {
            TS_079: DataSetEmpiar10453.getTestInterpTsDims(binningFactor=self.binFactor, nImgs=nImgs_079),
            TS_145: DataSetEmpiar10453.getTestInterpTsDims(binningFactor=self.binFactor, nImgs=nImgs_145),
        }
        testInterpAnglesCount = {
            TS_079: nImgs_079,
            TS_145: nImgs_145
        }
        # The refine tilt axis angles (updated to the rot angle) are different from expected in the test dataset
        # because of the excluded views
        testAcq = self.tsRefTAxAcqDict
        testAcq[TS_079].setTiltAxisAngle(85.4368)
        testAcq[TS_145].setTiltAxisAngle(84.7095)
        # Also, the interpolated TS min and max angles may have changed because of the image removal
        testInterpAcq = self.tsInterpDWAcqDict
        testInterpAcq[TS_079].setAngleMin(-54.01)
        testInterpAcq[TS_079].setAngleMax(56.99)
        testInterpAcq[TS_145].setAngleMin(-51.01)
        testInterpAcq[TS_145].setAngleMax(53.98)

        # Run the test
        importedTs = self._runImportTs()
        self._excludeTsSetViews(importedTs)  # Exclude some views at metadata level
        self._run_test_align_01(importedTs, expectedDimsInterpTs,
                                testAcq=testAcq,
                                testInterpAcq=testInterpAcq,
                                testInterpAnglesCount=testInterpAnglesCount,
                                eV=True)

    def _run_test_align_01(self, inTsSet, testInterpDims, testAcq=None, testDims=None,
                           testInterpAcq=None, testInterpAnglesCount=None, eV=False):
        msg = ("\n==> Testing AreTomo:"
               "\n\t- Align only"
               "\n\t- Generate also the interpolated TS"
               "\n\t- Dose weighting"
               "\n\t- No views are excluded by Aretomo"
               "\n\t- CTF not generated")
        eVLabel = ''
        if eV:
            msg += "\n\t- Input set of TS contains excluded views"
            eVLabel = ', eV'
        print(magentaStr(msg))

        if not testAcq:
            testAcq = self.tsRefTAxAcqDict
        if not testDims:
            testDims = self.expectedDimsTs
        if not testInterpAcq:
            testInterpAcq = self.tsInterpDWAcqDict
        if not testInterpAnglesCount:
            testInterpAnglesCount = self.nAngles

        # Run the protocol
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=inTsSet,
                                makeTomo=False,
                                doDW=True,
                                doEstimateCtf=False,
                                alignZ=self.alignZ,
                                binFactor=self.binFactor,
                                darkTol=0)
        prot.setObjLabel(f'Align only, no CTF{eVLabel}')
        self.launchProtocol(prot)

        # CHECK THE OUTPUTS
        # Tilt series
        excludedViews = self.excludedViewsDict if eV else None
        self.checkTiltSeries(getattr(prot, OUT_TS, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate,
                             expectedDimensions=testDims,
                             testAcqObj=testAcq,
                             hasAlignment=True,
                             alignment=ALIGN_2D,
                             anglesCount=self.nAngles,
                             excludedViewsDict=excludedViews)
        # Interpolated TS
        self.checkTiltSeries(getattr(prot, OUT_TS_ALN, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate * self.binFactor,
                             expectedDimensions=testInterpDims,
                             testAcqObj=testInterpAcq,
                             anglesCount=testInterpAnglesCount,
                             isInterpolated=True)
        # CTFs
        self.assertIsNone(getattr(prot, OUT_CTFS, None))


    def test_align_02(self):
        print(magentaStr("\n==> Testing AreTomo:"
                         "\n\t- Align only"
                         "\n\t- Generate also the interpolated TS"
                         "\n\t- No dose weighting"
                         "\n\t- Some views are excluded"
                         "\n\t- CTF generated"))

        # Expected values
        exludedViews = {TS_145: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                        TS_079: [40]}
        nAnglesDict = {TS_145: 27,
                       TS_079: 40}
        expectedDimsTsInterp = {TS_079: DataSetEmpiar10453.getTestInterpTsDims(binningFactor=self.binFactor,
                                                                               nImgs=nAnglesDict[TS_079]),
                                TS_145: DataSetEmpiar10453.getTestInterpTsDims(binningFactor=self.binFactor,
                                                                               nImgs=nAnglesDict[TS_145])}
        # Update the corresponding acquisition dictionary with the refined tilt axis angle values
        # Note: they're different from in the previous test because of the views exclusion
        tsAcqDict = self.tsAcqDict
        tsAcqDict[TS_079].setTiltAxisAngle(84.5478)
        tsAcqDict[TS_145].setTiltAxisAngle(84.0226)
        # Because some of the excluded views are at the end of the stack and DW is not applied in this test, the TS
        # accumulated values will be updated in the interpolated TS
        tsAcqDictInterp = self.tsInterpAcqDict
        acq079 = tsAcqDictInterp[TS_079]
        acq079.setAngleMax(56.99)

        acq145 = tsAcqDictInterp[TS_145]
        acq145.setAngleMin(-18.01)
        acq145.setAccumDose(138.15)

        tsAcqDictInterp = {TS_079: acq079,
                           TS_145:acq145}

        # Run the protocol
        importedTs = self._runImportTs()
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=importedTs,
                                makeTomo=False,
                                alignZ=self.alignZ,
                                binFactor=self.binFactor,
                                darkTol=0.8)
        prot.setObjLabel('Align only, exclude views')
        self.launchProtocol(prot)

        # CHECK THE OUTPUTS
        # Tilt series
        self.checkTiltSeries(getattr(prot, OUT_TS, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate,
                             expectedDimensions=self.expectedDimsTs,
                             testAcqObj=tsAcqDict,
                             hasAlignment=True,
                             alignment=ALIGN_2D,
                             anglesCount=self.nAngles,
                             excludedViewsDict=exludedViews)

        # Interpolated TS
        self.checkTiltSeries(getattr(prot, OUT_TS_ALN, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate * self.binFactor,
                             # Protocol sets the bin factor to 2 by default
                             expectedDimensions=expectedDimsTsInterp,
                             testAcqObj=tsAcqDictInterp,
                             anglesCount=nAnglesDict,
                             isInterpolated=True)
        # CTFs
        self._checkCTFs(getattr(prot, OUT_CTFS, None), excludedViewsDict=exludedViews)

    def test_alignAndReconstruct(self):
        print(magentaStr("\n==> Testing AreTomo:"
                         "\n\t- Align and reconstruct"
                         "\n\t- Interpolated TS not generated"
                         "\n\t- No dose weighting"
                         "\n\t- No views are excluded"
                         "\n\t- CTF generated"))

        # Run the protocol
        importedTs = self._runImportTs()
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=importedTs,
                                tomoThickness=self.unbinnedThk,
                                alignZ=self.alignZ,
                                binFactor=self.binFactor,
                                darkTol=0.1)
        prot.setObjLabel('Align & rec')
        self.launchProtocol(prot)

        # CHECK THE OUTPUTS
        # Tilt series
        self.checkTiltSeries(getattr(prot, OUT_TS, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate,
                             expectedDimensions=self.expectedDimsTs,
                             testAcqObj=self.tsRefTAxAcqDict,
                             hasAlignment=True,
                             alignment=ALIGN_2D,
                             anglesCount=self.nAngles)

        # CTFs
        self._checkCTFs(getattr(prot, OUT_CTFS, None))

        # Tomograms
        self.checkTomograms(getattr(prot, OUT_TOMO, None),
                            expectedSetSize=self.nTiltSeries,
                            expectedSRate=self.unbinnedSRate * self.binFactor,
                            expectedDimensions=self.expectedTomoDims,
                            expectedOriginShifts=self.expectedOriginShifts)

    def _checkCTFs(self, ctfSet, excludedViewsDict=None):
        self.checkCTFs(ctfSet,
                       expectedSetSize=self.nTiltSeries,
                       excludedViewsDict=excludedViewsDict,
                       expectedPsdFile=True)
