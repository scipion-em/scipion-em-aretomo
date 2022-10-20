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


def getTransformationMatrix(matrix):
    """ This method takes an IMOD-based transformation matrix (*.xf) and
    returns a 3D matrix containing the transformation matrices for each
    tilt-image belonging to the tilt-series. """

    numberLines = matrix.shape[0]
    frameMatrix = np.empty([3, 3, numberLines])

    for row in range(numberLines):
        frameMatrix[0, 0, row] = matrix[row][0]
        frameMatrix[1, 0, row] = matrix[row][2]
        frameMatrix[0, 1, row] = matrix[row][1]
        frameMatrix[1, 1, row] = matrix[row][3]
        frameMatrix[0, 2, row] = matrix[row][4]
        frameMatrix[1, 2, row] = matrix[row][5]
        frameMatrix[2, 0, row] = 0.0
        frameMatrix[2, 1, row] = 0.0
        frameMatrix[2, 2, row] = 1.0

    return frameMatrix


def readAlnFile(alignFn, newVersion=False):
    """ Read AreTomo output alignment file (.aln):
    aln2xf conversion taken from https://github.com/brisvag/stemia/blob/main/stemia/aretomo/aln2xf.py
    """
    # Read number of sections, as we need to ignore local alignments part of the file
    with open(alignFn) as f:
        f.readline()
        if newVersion:
            numSec = f.readline().strip("#").split()[-1]
        else:
            numSec = f.readline().strip("#").split()[0]

    data = np.loadtxt(alignFn, dtype=float, comments='#', max_rows=int(numSec))
    sec_nums = list(data[:, 0].astype(int))  # SEC
    tilt_angs = data[:, -1]  # TILT
    tilt_axes = data[:, 1]  # ROT
    angles = -np.radians(data[:, 1])  # ROT
    shifts = data[:, [3, 4]]  # TX, TY

    c, s = np.cos(angles), np.sin(angles)
    rot = np.empty((len(angles), 2, 2))
    rot[:, 0, 0] = c
    rot[:, 0, 1] = -s
    rot[:, 1, 0] = s
    rot[:, 1, 1] = c

    shifts_rot = np.einsum('ijk,ik->ij', rot, shifts)
    imod_matrix = np.concatenate([rot.reshape(-1, 4), -shifts_rot], axis=1)

    return sec_nums, imod_matrix, tilt_angs, tilt_axes
