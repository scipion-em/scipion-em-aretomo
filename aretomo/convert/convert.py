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

import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple, Type, Union

from pwem.emlib.image import ImageHandler

from tomo.objects import TiltSeries, TiltImage

# Number of columns in the AreTomo2 global alignment table:
# SEC ROT GMAG TX TY SMEAN SFIT SCALE BASE TILT
MIN_GLOBAL_COLUMNS = 10
# IMOD .xf line: a11 a12 a21 a22 dx dy
XF_LINE_FORMAT = "%9.3f %9.3f %9.3f %9.3f %9.2f %9.2f"
IDENTITY_XF = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def getTransformationMatrix(matrix: np.ndarray) -> np.ndarray:
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


class AretomoAln(NamedTuple):
    sections: List[int]
    imod_matrix: np.ndarray
    tilt_angles: List[float]
    tilt_axes: List[float]


class AlnParseError(ValueError):
    """ Raised when an AreTomo2 .aln file cannot be parsed. """


@dataclass
class GlobalAlign:
    """ One row of the AreTomo2 global alignment table. """
    sec: int        # section index into the raw tilt series (0-based)
    rot: float      # tilt-axis rotation angle, degrees
    tx: float       # X translation, unbinned pixels
    ty: float       # Y translation, unbinned pixels
    tilt: float     # nominal tilt angle, degrees


@dataclass
class DarkFrame:
    """ A tilt image AreTomo2 flagged as dark and excluded from alignment. """
    darkIdx: int    # index within the tilt-angle-sorted stack
    sec: int        # section index into the raw tilt series (0-based)
    tilt: float     # tilt angle, degrees


@dataclass
class AlnData:
    """ Parsed contents of an AreTomo2 .aln file. """
    globals: List[GlobalAlign] = field(default_factory=list)
    darkFrames: List[DarkFrame] = field(default_factory=list)
    rawSize: Optional[Tuple[int, int, int]] = None  # (nx, ny, nz)
    numPatches: int = 0

    @property
    def numRawSections(self) -> int:
        """ Total number of sections in the raw tilt series, falling back to
        the aligned + dark row count when the RawSize header is absent. """
        if self.rawSize is not None:
            return self.rawSize[2]
        return len(self.globals) + len(self.darkFrames)


def _parseRawSize(tokens) -> Tuple[int, int, int]:
    """ Parse the three integers of a '# RawSize = nx ny nz' header line. """
    ints = [int(float(t)) for t in tokens]
    if len(ints) != 3:
        raise AlnParseError(f"Expected 3 values for RawSize, got "
                            f"{len(ints)}: {list(tokens)}")
    return ints[0], ints[1], ints[2]


def _parseDarkFrame(body: str, lineno: int) -> DarkFrame:
    """ Parse a 'DarkFrame = darkIdx sec tilt' header line. """
    tokens = body.split("=", 1)[-1].split()
    if len(tokens) < 3:
        raise AlnParseError(f"Malformed DarkFrame entry on line "
                            f"{lineno}: {body!r}")
    try:
        return DarkFrame(darkIdx=int(float(tokens[0])),
                         sec=int(float(tokens[1])),
                         tilt=float(tokens[2]))
    except ValueError as exc:
        raise AlnParseError(f"Non-numeric DarkFrame values on line "
                            f"{lineno}: {body!r}") from exc


def _parseGlobalRow(line: str, lineno: int) -> GlobalAlign:
    """ Parse one data row of the global alignment table. """
    tokens = line.split()
    if len(tokens) < MIN_GLOBAL_COLUMNS:
        raise AlnParseError(f"Expected at least {MIN_GLOBAL_COLUMNS} columns "
                            f"on line {lineno}, got {len(tokens)}: {line!r}")
    try:
        values = [float(t) for t in tokens]
    except ValueError as exc:
        raise AlnParseError(f"Non-numeric value in global row on line "
                            f"{lineno}: {line!r}") from exc
    return GlobalAlign(sec=int(values[0]),   # SEC
                       rot=values[1],        # ROT (tilt axis, deg)
                       tx=values[3],         # TX
                       ty=values[4],         # TY
                       tilt=values[-1])      # TILT (last column)


def parseAlnFile(alignFn: Union[str, os.PathLike]) -> AlnData:
    """ Read and parse an AreTomo2 .aln file into an AlnData record, ignoring
    the local-alignment section. Raises AlnParseError on malformed content and
    FileNotFoundError when the file is missing. """
    aln = AlnData()
    inLocalSection = False

    try:
        handle = open(alignFn, "r")
    except OSError as exc:
        raise FileNotFoundError(f"Cannot open alignment file: "
                                f"{alignFn}") from exc

    with handle:
        for lineno, rawLine in enumerate(handle, start=1):
            line = rawLine.strip()
            if not line:
                continue

            if line.startswith("#"):
                # Header / comment line. Extract the fields we care about.
                body = line.lstrip("#").strip()
                lower = body.lower()
                if lower.startswith("rawsize"):
                    aln.rawSize = _parseRawSize(body.split("=", 1)[-1].split())
                elif lower.startswith("numpatches"):
                    try:
                        aln.numPatches = int(body.split("=", 1)[-1])
                    except ValueError:
                        aln.numPatches = 0
                elif lower.startswith("darkframe"):
                    aln.darkFrames.append(_parseDarkFrame(body, lineno))
                elif lower.startswith("local alignment"):
                    # Everything past this marker is per-patch data; stop.
                    inLocalSection = True
                continue

            if inLocalSection:
                continue

            aln.globals.append(_parseGlobalRow(line, lineno))

    if not aln.globals:
        raise AlnParseError(f"No global alignment rows found in {alignFn!r}; "
                            "is this a valid AreTomo2 .aln file?")
    return aln


def computeXf(rotDeg: np.ndarray, tx: np.ndarray, ty: np.ndarray) -> np.ndarray:
    """ Vectorised reproduction of AreTomo2's ImodUtil/CSaveXF.cpp. Returns an
    (n, 6) array with columns a11 a12 a21 a22 dx dy. The transform is the
    rotation R(-rot) with shift -R(-rot) . (tx, ty). """
    rotDeg = np.asarray(rotDeg, dtype=np.float64)
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    if not (rotDeg.shape == tx.shape == ty.shape):
        raise ValueError("rotDeg, tx and ty must share the same shape")

    negTheta = -np.radians(rotDeg)
    cosT, sinT = np.cos(negTheta), np.sin(negTheta)
    a11, a12, a21, a22 = cosT, -sinT, sinT, cosT
    dx = -(a11 * tx + a12 * ty)
    dy = -(a21 * tx + a22 * ty)

    return np.column_stack([a11, a12, a21, a22, dx, dy])


def alnToXf(aln: AlnData, fillDark: bool = True,
            order: str = "sec") -> np.ndarray:
    """ Convert parsed AlnData into an (n, 6) array of IMOD .xf rows.

    :param fillDark: when True (default) emit one row per raw section in
        section order, inserting identity transforms for dark frames (matches
        AreTomo2's -OutImod 1 output). When False only aligned rows are written.
    :param order: 'sec' to sort aligned rows by section index, 'file' to keep
        the order found in the file (ignored when fillDark is True).
    """
    if order not in ("sec", "file"):
        raise ValueError(f"order must be 'sec' or 'file', got {order!r}")

    rot = np.array([g.rot for g in aln.globals], dtype=np.float64)
    tx = np.array([g.tx for g in aln.globals], dtype=np.float64)
    ty = np.array([g.ty for g in aln.globals], dtype=np.float64)
    xf = computeXf(rot, tx, ty)

    if not fillDark:
        if order == "sec":
            sortIdx = np.argsort([g.sec for g in aln.globals], kind="stable")
            xf = xf[sortIdx]
        return xf

    # Raw-ordered output: place each aligned row at its section index and fill
    # the gaps (dark frames or otherwise missing sections) with the identity.
    nRaw = aln.numRawSections
    out = np.tile(np.asarray(IDENTITY_XF, dtype=np.float64), (nRaw, 1))
    for row, g in zip(xf, aln.globals):
        if not 0 <= g.sec < nRaw:
            raise ValueError(f"Section index {g.sec} out of range for "
                             f"RawSize nz={nRaw}")
        out[g.sec] = row
    return out


def readAlnFile(alignFn: Union[str, os.PathLike]) -> Type[AretomoAln]:
    """ Read AreTomo output alignment file (.aln) and populate AretomoAln with
    the aligned (non-dark) rows, in file order, plus their IMOD transforms.
    aln2xf conversion follows AreTomo2's ImodUtil/CSaveXF.cpp (originally after
    https://github.com/brisvag/stemia/blob/main/stemia/aretomo/aln2xf.py).
    """
    aln = parseAlnFile(alignFn)
    AretomoAln.sections = [g.sec for g in aln.globals]  # SEC
    AretomoAln.tilt_angles = np.array([g.tilt for g in aln.globals])  # TILT
    AretomoAln.tilt_axes = np.array([g.rot for g in aln.globals])  # ROT
    AretomoAln.imod_matrix = alnToXf(aln, fillDark=False, order="file")

    return AretomoAln


def writeAlnFile(ts: TiltSeries, tsFn: str, alignFn: Union[str, os.PathLike]):
    # xdim, ydim, zdim = ts.getDim()
    ih = ImageHandler()
    xdim, ydim, zdim, n = ih.getDimensions(tsFn)
    zdim = max(zdim, n)
    with open(alignFn, "w") as alnFile:
        alnFile.write("# AreTomo Alignment file generated by Scipion\n")
        alnFile.write(f"# RawSize = {xdim} {ydim} {zdim}\n")
        alnFile.write("# SEC     ROT         GMAG       TX          TY      "
                      "SMEAN     SFIT    SCALE     BASE     TILT\n")
        for ti in ts.iterItems(orderBy=TiltImage.TILT_ANGLE_FIELD):
            trMatrix = ti.getTransform().getMatrix()
            rot = np.rad2deg(np.arctan2(trMatrix[0, 1], trMatrix[0, 0]))
            # Shifts are rotated in AreTomo respecting Scipion's convention (see readAlnFile):
            rotMatrix = trMatrix[:2, :2]
            rotShifts = trMatrix[:2, 2]
            rotMatrixInv = np.linalg.inv(rotMatrix)
            unRotShifts = - np.dot(rotMatrixInv, rotShifts)
            alnFile.write(f"{ti.getIndex() - 1:>5}"
                          f"{rot:>11.4f}"
                          f"{1:>11.5f}"
                          f"{unRotShifts[0]:>11.3f}"
                          f"{unRotShifts[1]:>11.3f}"
                          f"{1:>9.2f}"
                          f"{1:>9.2f}"
                          f"{1:>9.2f}"
                          f"{0:>9.2f}"
                          f"{ti.getTiltAngle():>10.2f}\n")


def writeXfFile(matrix: np.ndarray, xfFn: Union[str, os.PathLike]):
    """ Write an (n, 6) transform array to an IMOD .xf text file. """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 6:
        raise ValueError(f"Expected a (n, 6) transform matrix, "
                         f"got shape {matrix.shape}")
    with open(xfFn, "w") as xfFile:
        for row in matrix:
            xfFile.write(XF_LINE_FORMAT % tuple(row))
            xfFile.write("\n")


def convertAlnToXf(alignFn: Union[str, os.PathLike],
                   xfFn: Union[str, os.PathLike],
                   fillDark: bool = True,
                   order: str = "sec") -> np.ndarray:
    """ Parse an AreTomo2 .aln file and write the corresponding IMOD .xf file.
    Returns the (n, 6) transform array that was written. See alnToXf for the
    meaning of fillDark and order. """
    aln = parseAlnFile(alignFn)
    xf = alnToXf(aln, fillDark=fillDark, order=order)
    writeXfFile(xf, xfFn)
    return xf


