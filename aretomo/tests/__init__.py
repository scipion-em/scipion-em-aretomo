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
from enum import Enum

from pyworkflow.tests import DataSet
from tomo.objects import TomoAcquisition

DataSet(name='tomo-em',
        folder='tomo-em',
        files={
            'ts1': 'tutorialData/BBa.st',
            'ts2': 'tutorialData/BBb.st'
        })

########################################################################################################################
EMDB_10453 = 'empiar_10453'

TS_079 = 'TS_079'
TS_145 = 'TS_145'
testAcq = TomoAcquisition(voltage=300,
                          sphericalAberration=2.7,
                          amplitudeContrast=0.1,
                          magnification=50000,
                          doseInitial=0,
                          tiltAxisAngle=84.8,
                          step=3)
# Acquisition of TS_079
testAcq079 = testAcq.clone()
testAcq079.setAngleMin(-60.005)
testAcq079.setAngleMax(59.9828)
testAcq079.setDosePerFrame(3.55679)
testAcq079.setAccumDose(145.8284)
# Acquisition of TS_079 - refined rot angle (overwrites the tilt axis angle)
testAcq079RefTAx = testAcq079.clone()
testAcq079RefTAx.setTiltAxisAngle(84.5276)
# Acquisition of TS_079 - Interpolated
testAcq079Interp = testAcq079.clone()
testAcq079Interp.setTiltAxisAngle(0.)
# Acquisition of TS_079 - Interpolated and dose-weighted
testAcq079InterpDW = testAcq079Interp.clone()
testAcq079InterpDW.setAccumDose(0.)
testAcq079InterpDW.setDoseInitial(0.)
testAcq079InterpDW.setDosePerFrame(0.)

# Acquisition of TS_145
testAcq145 = testAcq.clone()
testAcq145.setAngleMin(-60.007)
testAcq145.setAngleMax(59.9848)
testAcq145.setDosePerFrame(3.5424)
testAcq145.setAccumDose(145.2384)
# Acquisition of TS_145 - refined rot angle (overwrites the tilt axis angle)
testAcq145RefTAx = testAcq145.clone()
testAcq145RefTAx.setTiltAxisAngle(84.1236)
# Acquisition of TS_145 - Interpolated
testAcq145Interp = testAcq145.clone()
testAcq145Interp.setTiltAxisAngle(0.)
# Acquisition of TS_145 - Interpolated and dose-weighted
testAcq145InterpDW = testAcq145Interp.clone()
testAcq145InterpDW.setAccumDose(0.)
testAcq145InterpDW.setDoseInitial(0.)
testAcq145InterpDW.setDosePerFrame(0.)


class DataSetEmpiar10453(Enum):
    tsFilesPattern = '*.mdoc'
    unbinnedPixSize = 1.329
    tsDims = [5760, 4092, 41]
    nAngles = 41
    testAcq079 = testAcq079
    testAcq079RefTAx = testAcq079RefTAx
    testAcq079Interp = testAcq079Interp
    testAcq145 = testAcq145
    testAcq145RefTAx = testAcq145RefTAx
    testAcq145Interp = testAcq145Interp
    testTsAcqDict = {TS_079: testAcq079,
                     TS_145: testAcq145}
    testTsRefTAxAcqDict = {TS_079: testAcq079RefTAx,
                           TS_145: testAcq145RefTAx}
    testTsInterpAcqDict = {TS_079: testAcq079Interp,
                           TS_145: testAcq145Interp}
    testTsInterpDWAcqDict = {TS_079: testAcq079InterpDW,
                             TS_145: testAcq145InterpDW}

    @classmethod
    def getTestTsDims(cls, binningFactor=1, nImgs=-1):
        tsDims = cls.tsDims.value
        dim1 = int(tsDims[0] / binningFactor)
        dim2 = int(tsDims[1] / binningFactor)
        return [dim1 if dim1 % 2 == 0 else dim1 - 1,  # If odd, aretomo subtracts 1
                dim2 if dim2 % 2 == 0 else dim2 - 1,
                int(nImgs)]

    @classmethod
    def getTestInterpTsDims(cls, binningFactor=1, nImgs=-1):
        newDims = cls.getTestTsDims(binningFactor=binningFactor, nImgs=nImgs)
        return [newDims[1], newDims[0], newDims[2]]


DataSet(name=EMDB_10453, folder=EMDB_10453, files={el.name: el.value for el in DataSetEmpiar10453})
