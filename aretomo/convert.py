# **************************************************************************
# *
# * Authors:     Federico P. de Isidro Gomez (fp.deisidro@cnb.csic.es) [1]
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
# **************************************************************************

import numpy as np


def getTransformationMatrix(matrixFile):
    """ This method takes an IMOD-based transformation matrix file (*.xf) path and
    returns a 3D matrix containing the transformation matrices for each
    tilt-image belonging to the tilt-series. """

    with open(matrixFile, "r") as matrix:
        lines = matrix.readlines()

    numberLines = len(lines)
    frameMatrix = np.empty([3, 3, numberLines])

    i = 0
    for line in lines:
        values = line.split()
        frameMatrix[0, 0, i] = float(values[0])
        frameMatrix[1, 0, i] = float(values[2])
        frameMatrix[0, 1, i] = float(values[1])
        frameMatrix[1, 1, i] = float(values[3])
        frameMatrix[0, 2, i] = float(values[4])
        frameMatrix[1, 2, i] = float(values[5])
        frameMatrix[2, 0, i] = 0.0
        frameMatrix[2, 1, i] = 0.0
        frameMatrix[2, 2, i] = 1.0
        i += 1

    return frameMatrix
