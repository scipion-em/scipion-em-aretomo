# **************************************************************************
# *
# * Authors:     Scipion Team (scipion@cnb.csic.es) [1]
# *
# * [1] National Center of Biotechnology, CSIC, Spain
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

import os.path
import pyworkflow.utils as pwutils 
from pwem.objects import CTFModel, SetOfParticles


with pwutils.weakImport('tomo'):
    from tomo.objects import CTFTomo


class AretomoCtfParser:
    """ Import CTF estimated with Aretomo. """
    # def __init__(self, protocol):
    #     self.protocol = protocol
    #     self.copyOrLink = self.protocol.getCopyOrLink()

    # def importCTF(self, mic, fileName):
    #     """ Create a CTF model and populate its values.
    #     :param mic: input micrograph object
    #     :param fileName: input file to be parsed
    #     :return: CTFModel object
    #     """
    #     ctf = CTFModel()
    #     ctf.setMicrograph(mic)
    #     readCtfModel(ctf, fileName)
    #
    #     fnBase = pwutils.removeExt(fileName)
    #     psdFile = self._findPsdFile(fnBase)
    #     ctf.setPsdFile(psdFile)
    #
    #     return ctf

    def parseTSDefocusFile(self, ts, fileName, output):
        pass
    #     """ Parse tilt-series ctf estimation file.
    #     :param ts: input tilt-series
    #     :param fileName: input file to be parsed
    #     :param output: output CTFTomoSeries
    #     """
    #     tsId = ts.getTsId()
    #     fnBase = os.path.join(os.path.dirname(fileName), tsId)
    #     outputPsd = self._findPsdFile(fnBase)
    #     ctfResult = parseCtffind4Output(fileName)
    #     ctf = CTFModel()
    #
    #     for i, ti in enumerate(ts):
    #         if ti.isEnabled():
    #             self.getCtfTi(ctf, ctfResult, i, outputPsd)
    #         else:
    #             ctf.setStandardDefocus(0, 0, 0)
    #         newCtfTomo = CTFTomo.ctfModelToCtfTomo(ctf)
    #         newCtfTomo.setIndex(i + 1)
    #         output.append(newCtfTomo)
    #
    #     output.calculateDefocusUDeviation()
    #     output.calculateDefocusVDeviation()
    #
    # @staticmethod
    # def getCtfTi(ctf, ctfArray, tiIndex, psdStack=None):
    #     """ Parse the CTF object estimated for this Tilt-Image. """
    #     readCtfModelStack(ctf, ctfArray, item=tiIndex)
    #     if psdStack is not None:
    #         ctf.setPsdFile(f"{tiIndex + 1}@" + psdStack)
    #
    # @staticmethod
    # def _findPsdFile(fnBase):
    #     """ Try to find the given PSD file associated with the cttfind log file
    #     We handle special cases of .ctf extension and _ctffind4 prefix for Relion runs
    #     """
    #     for suffix in ['_psd.mrc', '.mrc', '_ctf.mrcs',
    #                    '.mrcs', '.ctf']:
    #         psdPrefixes = [fnBase,
    #                        fnBase.replace('_ctffind4', '')]
    #         for prefix in psdPrefixes:
    #             psdFile = prefix + suffix
    #             if os.path.exists(psdFile):
    #                 if psdFile.endswith('.ctf'):
    #                     psdFile += ':mrc'
    #                 return psdFile
    #     return None


