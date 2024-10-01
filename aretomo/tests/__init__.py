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
testAcq079.setAccumDose(138.71481)
# Acquisition of TS_145
testAcq145 = testAcq.clone()
testAcq145.setAngleMin(-60.007)
testAcq145.setAngleMax(59.9848)
testAcq145.setDosePerFrame(3.5424)
testAcq145.setAccumDose(138.1536)


class DataSetEmpiar10453(Enum):
    unbinnedPixSize = 1.33
    tsDims = [5760, 4092, 41]
    nAngles = 41
    testAcqDict = {TS_079: testAcq079,
                   TS_145: testAcq145}

    @classmethod
    def getInterpTsDims(cls, binningFactor=1, nImgs=-1):
        return [cls.tsDims[0] / binningFactor,
                cls.tsDims[1] / binningFactor,
                nImgs]

DataSet(name=EMDB_10453, folder=EMDB_10453, files={el.name: el.value for el in DataSetEmpiar10453})
