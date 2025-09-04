# **************************************************************************
# *
# * Authors:     Scipion Team (scipion@cnb.csic.es) [1]
# *
# * [1] Centro Nacional de Biotecnologia, CSIC, Spain
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
# *****************************************************************************
import copy
from typing import Union
import numpy as np
from imod.constants import OUTPUT_TILTSERIES_NAME
from imod.protocols import ProtImodImportTransformationMatrix
from pwem import ALIGN_2D
from pyworkflow.utils import magentaStr, blueStr
from pyworkflow.tests import DataSet, setupTestProject
from tomo.objects import SetOfTiltSeries, SetOfCTFTomoSeries, TiltSeries, SetOfTomograms
from tomo.protocols import ProtImportTs
from tomo.tests import RE4_STA_TUTO, DataSetRe4STATuto
from tomo.tests.test_base_centralized_layer import TestBaseCentralizedLayer
from . import DataSetEmpiar10453, EMDB_10453, TS_079, TS_145
from ..protocols.protocol_aretomo import (ProtAreTomoAlignRecon, OUT_TS, OUT_TOMO,
                                          OUT_CTFS)


class TestAreTomo2Base(TestBaseCentralizedLayer):
    ds = None
    unbinnedSRate = DataSetEmpiar10453.unbinnedPixSize.value
    nAngles = DataSetEmpiar10453.nAngles.value
    tsAcqDict = DataSetEmpiar10453.testTsAcqDict.value
    tsRefTAxAcqDict = DataSetEmpiar10453.testTsRefTAxAcqDict.value
    nTiltSeries = len(tsAcqDict)

    alignZ = 800
    binFactor = 4
    unbinnedThk = 1200

    expectedTomoDims = [1022, 1440, 300]

    excludedViewsDict = {
        TS_079: [0, 1, 38, 40],
        TS_145: [0, 1, 2, 39, 40]
    }
    # Because of AreTomo's slightly different angular refinement results depending on the Cuda toolkit version, etc,
    # these tests' angular tolerance for both the tilt angles and the tilt axis / rotation angle are 1 degree instead
    # of the default 0.1 deg.
    tiltAnglesTolDeg = 1
    rotAngleTolDeg = 1

    @classmethod
    def setUpClass(cls):
        print(blueStr('===> TESTS INFORMATION <==='
                      '\nTests originally designed and passing with Aretomo2-1-1-3, Ubuntu 22.04, cuda-12.1'))
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
        importedTs = self._runImportTs()
        self._run_test_align_01(importedTs)

    def test_align_01_excluded_views(self):
        # The refine tilt axis angles (updated to the rot angle) are different from expected in the test dataset
        # because of the excluded views
        testAcq = copy.deepcopy(self.tsRefTAxAcqDict)
        testAcq[TS_079].setTiltAxisAngle(85.4368)
        testAcq[TS_145].setTiltAxisAngle(84.7095)

        # Run the test
        importedTs = self._runImportTs()
        self._excludeTsSetViews(importedTs)  # Exclude some views at metadata level
        self._run_test_align_01(importedTs,
                                testAcq=testAcq,
                                eV=True)

    def _run_test_align_01(self,
                           inTsSet: SetOfTiltSeries,
                           testAcq: dict = None,
                           testDims: dict = None,
                           eV: bool = False) -> None:
        msg = ("\n==> Testing AreTomo:"
               "\n\t- Align only"
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

        # Run the protocol
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=inTsSet,
                                makeTomo=False,
                                doDW=True,
                                doEstimateCtf=False,
                                alignZ=self.alignZ,
                                binFactor=self.binFactor,
                                darkTol=0)
        prot.setObjLabel(f'Align only, no CTF, DW{eVLabel}')
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
                             excludedViewsDict=excludedViews,
                             tiltAnglesTolDeg=self.tiltAnglesTolDeg,
                             rotAngleTolDeg=self.rotAngleTolDeg)
        # CTFs
        self.assertIsNone(getattr(prot, OUT_CTFS, None))

    def test_align_02(self):
        # Expected values
        excludedViews = {TS_145: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                         TS_079: [40]}

        # Update the corresponding acquisition dictionary with the refined tilt axis angle values
        # Note: they're different from in the previous test because of the views exclusion
        tsAcqDict = copy.deepcopy(self.tsAcqDict)
        tsAcqDict[TS_079].setTiltAxisAngle(84.5478)
        tsAcqDict[TS_145].setTiltAxisAngle(84.0226)

        # Run the test
        importedTs = self._runImportTs()
        self._run_test_align_02(inTsSet=importedTs,
                                testAcq=tsAcqDict,
                                excludedViews=excludedViews)

    def test_align_02_excluded_views(self):
        """The input TS have some excluded views, and AreTomo's dark tolerance threshold will remove some
        views, too:
            - TS_079:
                * Input views marked to be excluded: [0, 1, 38, 40]
                * Views that will be removed by AreTomo: []
            - TS_145:
                * Input views marked to be excluded: [0, 1, 2, 39, 40]
                * Views that will be removed by AreTomo, referred to:
                    -> Re-stacked TS: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                    -> Input TS: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        """
        # Expected values
        excludedViews = {TS_145: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 39, 40],
                         TS_079: [0, 1, 38, 40]}
        nImgs_079 = self.nAngles - len(excludedViews[TS_079])
        nImgs_145 = self.nAngles - len(excludedViews[TS_145])
        nAnglesDict = {TS_145: nImgs_145,
                       TS_079: nImgs_079}
        # Update the corresponding acquisition dictionary with the refined tilt axis angle values
        # Note: they're different from in the previous test because of the views exclusion
        tsAcqDict = copy.deepcopy(self.tsAcqDict)
        tsAcqDict[TS_079].setTiltAxisAngle(85.4368)
        tsAcqDict[TS_145].setTiltAxisAngle(84.1337)

        # Run the test
        importedTs = self._runImportTs()
        self._excludeTsSetViews(importedTs)  # Exclude some views at metadata level
        self._run_test_align_02(inTsSet=importedTs,
                                testAcq=tsAcqDict,
                                excludedViews=excludedViews,
                                eV=True)

    def _run_test_align_02(self,
                           inTsSet: SetOfTiltSeries,
                           testAcq: dict = None,
                           excludedViews: dict = None,
                           eV: bool = False):
        msg = ("\n==> Testing AreTomo:"
               "\n\t- Align only"
               "\n\t- No dose weighting"
               "\n\t- Some views are excluded"
               "\n\t- CTF generated")
        eVLabel = ''
        if eV:
            msg += "\n\t- Input set of TS contains excluded views"
            eVLabel = ', eV'
        print(magentaStr(msg))

        # Run the protocol
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=inTsSet,
                                makeTomo=False,
                                alignZ=self.alignZ,
                                binFactor=self.binFactor,
                                darkTol=0.8)
        prot.setObjLabel(f'Align only, exclude views{eVLabel}')
        self.launchProtocol(prot)

        # CHECK THE OUTPUTS
        # Tilt series
        self.checkTiltSeries(getattr(prot, OUT_TS, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate,
                             expectedDimensions=self.expectedDimsTs,
                             testAcqObj=testAcq,
                             hasAlignment=True,
                             alignment=ALIGN_2D,
                             anglesCount=self.nAngles,
                             excludedViewsDict=excludedViews,
                             tiltAnglesTolDeg=self.tiltAnglesTolDeg,
                             rotAngleTolDeg=self.rotAngleTolDeg)

        # CTFs
        self._checkCTFs(getattr(prot, OUT_CTFS, None), excludedViewsDict=excludedViews)

    def test_alignAndReconstruct(self):
        # Run the protocol
        importedTs = self._runImportTs()
        self._run_test_alignAndReconstruct(importedTs)

    def test_alignAndReconstruct_excluded_views(self):
        # Expected values
        # The refine tilt axis angles (updated to the rot angle) are different from expected in the test dataset
        # because of the excluded views
        testAcq = copy.deepcopy(self.tsRefTAxAcqDict)
        testAcq[TS_079].setTiltAxisAngle(85.4368)
        testAcq[TS_145].setTiltAxisAngle(84.7095)

        # Run the protocol
        importedTs = self._runImportTs()
        self._excludeTsSetViews(importedTs)  # Exclude some views at metadata level
        self._run_test_alignAndReconstruct(importedTs, testAcq=testAcq, eV=True)

    def _checkCTFs(self,
                   ctfSet: SetOfCTFTomoSeries,
                   excludedViewsDict: dict = None):
        self.checkCTFs(ctfSet,
                       expectedSetSize=self.nTiltSeries,
                       excludedViewsDict=excludedViewsDict,
                       expectedPsdFile=True)

    def _run_test_alignAndReconstruct(self,
                                      inTsSet: SetOfTiltSeries,
                                      testAcq: dict = None,
                                      eV: bool = False) -> None:
        msg = ("\n==> Testing AreTomo:"
               "\n\t- Align and reconstruct"
               "\n\t- No dose weighting"
               "\n\t- No views are excluded"
               "\n\t- CTF generated")
        eVLabel = ''
        if eV:
            msg += "\n\t- Input set of TS contains excluded views"
            eVLabel = ', eV'
        print(magentaStr(msg))

        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=inTsSet,
                                tomoThickness=self.unbinnedThk,
                                alignZ=self.alignZ,
                                binFactor=self.binFactor,
                                darkTol=0.1)
        prot.setObjLabel(f'Align & rec{eVLabel}')
        self.launchProtocol(prot)

        # CHECK THE OUTPUTS
        # Tilt series
        excludedViews = self.excludedViewsDict if eV else None
        expectedAcq = testAcq if testAcq else self.tsRefTAxAcqDict
        self.checkTiltSeries(getattr(prot, OUT_TS, None),
                             expectedSetSize=self.nTiltSeries,
                             expectedSRate=self.unbinnedSRate,
                             expectedDimensions=self.expectedDimsTs,
                             testAcqObj=expectedAcq,
                             hasAlignment=True,
                             alignment=ALIGN_2D,
                             anglesCount=self.nAngles,
                             excludedViewsDict=excludedViews,
                             tiltAnglesTolDeg=self.tiltAnglesTolDeg,
                             rotAngleTolDeg=self.rotAngleTolDeg)

        # CTFs
        self._checkCTFs(getattr(prot, OUT_CTFS, None), excludedViewsDict=excludedViews)

        # Tomograms
        self.checkTomograms(getattr(prot, OUT_TOMO, None),
                            expectedSetSize=self.nTiltSeries,
                            expectedSRate=self.unbinnedSRate * self.binFactor,
                            expectedDimensions=self.expectedTomoDims,
                            expectedOriginShifts=self.expectedOriginShifts)

class TestAretomo2OnlyRec(TestBaseCentralizedLayer):
    unbinnedSRate = DataSetRe4STATuto.unbinnedPixSize.value
    binFactor = 4
    unbinnedThk = 1200
    expectedDims = [958, 926, 300]

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet(RE4_STA_TUTO)
        cls._runPreviousProtocols()

    @classmethod
    def _runPreviousProtocols(cls):
        cls.importedTs = cls._runImportTs()

    @classmethod
    def _runImportTs(cls):
        print(magentaStr("\n==> Importing the tilt series:"))
        protImportTs = cls.newProtocol(ProtImportTs,
                                       filesPath=cls.ds.getFile(DataSetRe4STATuto.tsPath.value),
                                       filesPattern=DataSetRe4STATuto.tsPattern.value,
                                       exclusionWords=DataSetRe4STATuto.exclusionWordsTs03ts54.value,
                                       anglesFrom=2,  # From tlt file
                                       voltage=DataSetRe4STATuto.voltage.value,
                                       magnification=DataSetRe4STATuto.magnification.value,
                                       sphericalAberration=DataSetRe4STATuto.sphericalAb.value,
                                       amplitudeContrast=DataSetRe4STATuto.amplitudeContrast.value,
                                       samplingRate=cls.unbinnedSRate,
                                       doseInitial=DataSetRe4STATuto.initialDose.value,
                                       dosePerFrame=DataSetRe4STATuto.dosePerTiltImg.value,
                                       tiltAxisAngle=DataSetRe4STATuto.tiltAxisAngle.value)

        cls.launchProtocol(protImportTs)
        tsImported = getattr(protImportTs, protImportTs.OUTPUT_NAME, None)
        return tsImported

    @classmethod
    def _runImportTrMatrix(cls):
        print(magentaStr("\n==> Importing the TS' transformation matrices with IMOD:"))
        protImportTrMatrix = cls.newProtocol(ProtImodImportTransformationMatrix,
                                             filesPath=cls.ds.getFile(DataSetRe4STATuto.tsPath.value),
                                             filesPattern=DataSetRe4STATuto.transformPattern.value,
                                             inputSetOfTiltSeries=cls.importedTs)
        cls.launchProtocol(protImportTrMatrix)
        outTsSet = getattr(protImportTrMatrix, OUTPUT_TILTSERIES_NAME, None)
        return outTsSet

    @classmethod
    def _excludeTsSetViews(cls,
                           tsSet: SetOfTiltSeries,
                           excludedViewsDict: dict) -> None:
        tsList = [ts.clone(ignoreAttrs=[]) for ts in tsSet]
        for ts in tsList:
            cls._excludeTsViews(tsSet, ts, excludedViewsDict[ts.getTsId()])

    @staticmethod
    def _excludeTsViews(tsSet: SetOfTiltSeries,
                        ts: TiltSeries,
                        excludedViewsList: dict) -> None:
        tiList = [ti.clone() for ti in ts]
        for i, ti in enumerate(tiList):
            if i in excludedViewsList:
                ti._objEnabled = False
                ts.update(ti)
        ts.write()
        tsSet.update(ts)
        tsSet.write()

    def _runAreTomoRec(self,
                       inTsSet: SetOfTiltSeries,
                       eV: bool = False) -> Union[SetOfTomograms, None]:
        msg = "\n==> Testing AreTomo (only tomogram reconstruction):"
        eVLabel = ''
        if eV:
            msg += "\n\t- Input set of TS contains excluded views"
            eVLabel = ', eV'
        print(magentaStr(msg))

        # Run the protocol
        prot = self.newProtocol(ProtAreTomoAlignRecon,
                                inputSetOfTiltSeries=inTsSet,
                                skipAlign=True,
                                makeTomo=True,
                                tomoThickness=1200,
                                binFactor=self.binFactor)
        prot.setObjLabel(f'Reconstruct only {eVLabel}')
        self.launchProtocol(prot)
        return getattr(prot, OUT_TOMO, None)

    def _checkResults(self,
                      inTomoSet: SetOfTomograms) -> None:
        self.checkTomograms(inTomoSet,
                            expectedSetSize=2,
                            expectedSRate=self.unbinnedSRate * self.binFactor,
                            expectedDimensions=self.expectedDims)

    def test_rec(self) -> None:
        tsWithAlignment = self._runImportTrMatrix()
        # Run the protocol
        tomoSet = self._runAreTomoRec(tsWithAlignment)
        # Check the results
        self._checkResults(tomoSet)

    def test_rec_eV(self) -> None:
        TS_03 = 'TS_03'
        TS_54 = 'TS_54'
        excludedViewsDict = {
            TS_03: [0, 1, 2, 38, 39],
            TS_54: [0, 1, 38, 39, 40]
        }
        # Exclude some views at metadata level
        tsWithAlignment = self._runImportTrMatrix()
        self._excludeTsSetViews(tsWithAlignment, excludedViewsDict)
        # Run the protocol
        tomoSet = self._runAreTomoRec(tsWithAlignment, eV=True)
        # Check the results
        self._checkResults(tomoSet)