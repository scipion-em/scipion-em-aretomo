# **************************************************************************
# *
# * Authors:     Grigory Sharov (gsharov@mrc-lmb.cam.ac.uk)
# *              Scipion Team (scipion@cnb.csic.es) [1]
# *
# * MRC Laboratory of Molecular Biology (MRC-LMB)
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
import tempfile
from os.path import join

import pwem
import pyworkflow.utils as pwutils
from aretomo.constants import ARETOMO_HOME, ARETOMO_CUDA_LIB, V1_1_3, DEFAULT_VERSION
from pyworkflow import VarTypes, TOMO


__version__ = '4.0.0'
_logo = "aretomo_logo.png"
_references = ['Zheng2022']


class Plugin(pwem.Plugin):
    _homeVar = ARETOMO_HOME
    _pathVars = [ARETOMO_HOME, ARETOMO_CUDA_LIB]
    _supportedVersions = [V1_1_3]
    _url = "https://github.com/scipion-em/scipion-em-aretomo"
    _processingField = [TOMO]

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(ARETOMO_HOME, f'aretomo2-{DEFAULT_VERSION}',
                         description="Root folder where aretomo was extracted. Is assumes "
                                     "binaries are under that folder/bin.",
                         var_type=VarTypes.FOLDER)
        cls._defineVar(ARETOMO_CUDA_LIB, pwem.Config.CUDA_LIB,
                       description="Path to the CUDA lib path to use with the Aretomo binary.",
                       var_type=VarTypes.FOLDER)

    @classmethod
    def getEnviron(cls):
        """ Return the environment to run AreTomo. """
        environ = pwutils.Environ(os.environ)
        # Get AreTomo CUDA library path if defined
        cudaLib = cls.getVar(ARETOMO_CUDA_LIB, pwem.Config.CUDA_LIB)
        environ.addLibrary(cudaLib)

        return environ

    @classmethod
    def getProgram(cls) -> str:
        """ Return the program binary that will be used. """
        return cls.getHome('AreTomo2')

    @classmethod
    def defineBinaries(cls, env):
        ARETOMO_INSTALLED = 'aretomo_installed'
        MAKEFILE11 = 'makefile11'
        aretomoHome = cls.getVar(ARETOMO_HOME)
        fixedMakeFile = cls.__createFatBinaryMakeFile()
        cmd = [
            f'cd .. && rm -rf {aretomoHome} && '
            f'git clone https://github.com/czimaginginstitute/AreTomo2.git {aretomoHome} && '
            f'cd {aretomoHome} && '
            'git checkout main && '
            f'rm {MAKEFILE11} && '  # remove the original makefile
            f'mv {fixedMakeFile} {join(aretomoHome, MAKEFILE11)} && '  # move the fixed makefile to the installation folder
            f'make exe -f {MAKEFILE11} && '
            f'touch {ARETOMO_INSTALLED}'
        ]

        installationCmds = [
            (cmd, ARETOMO_INSTALLED)
        ]

        env.addPackage('aretomo2', version=DEFAULT_VERSION,
                        tar = 'void.tgz',
                        neededProgs = ['git', 'gcc', 'g++', 'make', 'cmake'],
                        commands = installationCmds,
                        updateCuda = True,
                        default = True)

    @classmethod
    def __createFatBinaryMakeFile(cls) -> str:
        """
        Creates a fixed Makefile11 file that will replace during installation time
        the original AreTomo2 Makefile11 with an optimized version that includes
        Fat Binary support, disables PIE, and adds fPIC.
        """
        # Define the contents of the file.
        # NOTE: Explicit tab characters (\t) are used in the recipe commands to comply with the Make standard.
        contents = (
            "#-------------------------------------------------------------------------------\n"
            "# 1. CUDAHOME is now a conditional variable (?=). If not defined in the \n"
            "#    environment, it defaults to the standard CUDA symlink in Ubuntu.\n"
            "# 2. libutil.a, libmrcfile.a are in LibSrc folder where the source code is\n"
            "#    also provided in Util and Mrcfile. To recompile them, run \"make clean\"\n"
            "#    followed by \"make all\" in Util first and then Mrcfile next.\n"
            "#-------------------------------------------------------------------------------\n"
            "CUDAHOME ?= /usr/local/cuda\n"
            "PRJHOME = $(shell pwd)\n"
            "CUDAINC = $(CUDAHOME)/include\n"
            "CUDALIB = $(CUDAHOME)/lib64\n"
            "PRJINC = $(PRJHOME)/LibSrc/Include\n"
            "PRJLIB = $(PRJHOME)/LibSrc/Lib\n"
            "\n"
            "#-------------------\n"
            "CUSRCS = ./Util/GAddImages.cu \\\n"
            "         ./Util/GFFT1D.cu \\\n"
            "         ./Util/GFFT2D.cu \\\n"
            "         ./Util/GFFTUtil2D.cu \\\n"
            "         ./Util/GNormalize2D.cu \\\n"
            "         ./Util/GRoundEdge.cu \\\n"
            "         ./Util/GRoundEdge1D.cu \\\n"
            "         ./Util/GThreshold2D.cu \\\n"
            "         ./Util/GStretch.cu \\\n"
            "         ./Util/GXcf2D.cu \\\n"
            "         ./Util/GRotate2D.cu \\\n"
            "         ./Util/GShiftRotate2D.cu \\\n"
            "         ./Util/GRemoveSpikes2D.cu \\\n"
            "         ./Util/GBinImage2D.cu \\\n"
            "         ./Util/GCC1D.cu \\\n"
            "         ./Util/GCC2D.cu \\\n"
            "         ./Util/GRealCC2D.cu \\\n"
            "         ./Util/GCalcMoment2D.cu \\\n"
            "         ./Util/GFindMinMax2D.cu \\\n"
            "         ./Util/GFourierCrop2D.cu \\\n"
            "         ./Util/GMutualMask2D.cu \\\n"
            "         ./Util/GPositivity2D.cu \\\n"
            "         ./Util/GCorrLinearInterp.cu \\\n"
            "         ./FindCtf/GCalcCTF1D.cu \\\n"
            "         ./FindCtf/GCalcCTF2D.cu \\\n"
            "         ./FindCtf/GCalcSpectrum.cu \\\n"
            "         ./FindCtf/GCC1D.cu \\\n"
            "         ./FindCtf/GCC2D.cu \\\n"
            "         ./FindCtf/GRadialAvg.cu \\\n"
            "         ./FindCtf/GRemoveMean.cu \\\n"
            "         ./FindCtf/GRmBackground2D.cu \\\n"
            "         ./FindCtf/GRoundEdge.cu \\\n"
            "         ./FindCtf/GLowpass2D.cu \\\n"
            "         ./CommonLine/GFunctions.cu \\\n"
            "         ./CommonLine/GCalcCommonRegion.cu \\\n"
            "         ./CommonLine/GGenCommonLine.cu \\\n"
            "         ./CommonLine/GRemoveMean.cu \\\n"
            "         ./CommonLine/GInterpolateLineSet.cu \\\n"
            "         ./ProjAlign/GReproj.cu \\\n"
            "         ./ProjAlign/GProjXcf.cu \\\n"
            "         ./DoseWeight/GDoseWeightImage.cu \\\n"
            "         ./Recon/GRWeight.cu \\\n"
            "         ./Recon/GBackProj.cu \\\n"
            "         ./Recon/GForProj.cu \\\n"
            "         ./Recon/GDiffProj.cu \\\n"
            "         ./Recon/GWeightProjs.cu \\\n"
            "         ./Massnorm/GFlipInt2D.cu \\\n"
            "         ./Massnorm/GPositivity.cu \\\n"
            "         ./Correct/GCorrPatchShift.cu \\\n"
            "         ./PatchAlign/GAddImages.cu \\\n"
            "         ./PatchAlign/GRandom2D.cu \\\n"
            "         ./PatchAlign/GExtractPatch.cu \\\n"
            "         ./PatchAlign/GCommonArea.cu \\\n"
            "         ./PatchAlign/GGenXcfImage.cu \\\n"
            "         ./PatchAlign/GPartialCopy.cu \\\n"
            "         ./PatchAlign/GNormByStd2D.cu\n"
            "\n"
            "CUCPPS = $(patsubst %.cu, %.cpp, $(CUSRCS))\n"
            "#------------------------------------------\n"
            "SRCS = ./Util/CNextItem.cpp \\\n"
            "       ./Util/CSplitItems.cpp \\\n"
            "       ./Util/CSimpleFuncs.cpp \\\n"
            "       ./Util/CParseArgs.cpp \\\n"
            "       ./Util/CFileName.cpp \\\n"
            "       ./Util/CRemoveSpikes1D.cpp \\\n"
            "       ./Util/CPad2D.cpp \\\n"
            "       ./Util/CPeak2D.cpp \\\n"
            "       ./Util/CTRDecompose2D.cpp \\\n"
            "       ./Util/CSaveTempMrc.cpp \\\n"
            "       ./Util/CStrLinkedList.cpp \\\n"
            "       ./Util/CReadDataFile.cpp \\\n"
            "       ./MrcUtil/CTomoStack.cpp \\\n"
            "       ./MrcUtil/CAlignParam.cpp \\\n"
            "       ./MrcUtil/CLocalAlignParam.cpp \\\n"
            "       ./MrcUtil/CPatchShifts.cpp \\\n"
            "       ./MrcUtil/CSaveAlnFile.cpp \\\n"
            "       ./MrcUtil/CLoadAlnFile.cpp \\\n"
            "       ./MrcUtil/CLoadAngFile.cpp \\\n"
            "       ./MrcUtil/CRemoveDarkFrames.cpp \\\n"
            "       ./MrcUtil/CCalcStackStats.cpp \\\n"
            "       ./MrcUtil/CDarkFrames.cpp \\\n"
            "       ./MrcUtil/CLoadStack.cpp \\\n"
            "       ./MrcUtil/CLoadMain.cpp \\\n"
            "       ./MrcUtil/CSaveStack.cpp \\\n"
            "       ./MrcUtil/CGenCentralSlices.cpp \\\n"
            "       ./MrcUtil/CCropVolume.cpp \\\n"
            "       ./ImodUtil/CSaveXF.cpp \\\n"
            "       ./ImodUtil/CSaveTilts.cpp \\\n"
            "       ./ImodUtil/CSaveCsv.cpp \\\n"
            "       ./ImodUtil/CSaveXtilts.cpp \\\n"
            "       ./ImodUtil/CImodUtil.cpp \\\n"
            "       ./FindCtf/CCtfResults.cpp \\\n"
            "       ./FindCtf/CCtfTheory.cpp \\\n"
            "       ./FindCtf/CFindCtf1D.cpp \\\n"
            "       ./FindCtf/CFindCtf2D.cpp \\\n"
            "       ./FindCtf/CFindCtfBase.cpp \\\n"
            "       ./FindCtf/CFindCtfHelp.cpp \\\n"
            "       ./FindCtf/CFindDefocus1D.cpp \\\n"
            "       ./FindCtf/CFindDefocus2D.cpp \\\n"
            "       ./FindCtf/CGenAvgSpectrum.cpp \\\n"
            "       ./FindCtf/CSpectrumImage.cpp \\\n"
            "       ./FindCtf/CSaveCtfResults.cpp \\\n"
            "       ./FindCtf/CFindCtfMain.cpp \\\n"
            "       ./ProjAlign/CCentralXcf.cpp \\\n"
            "       ./ProjAlign/CParam.cpp \\\n"
            "       ./ProjAlign/CRemoveSpikes.cpp \\\n"
            "       ./ProjAlign/CCalcReproj.cpp \\\n"
            "       ./ProjAlign/CProjAlignMain.cpp \\\n"
            "       ./StreAlign/CStretchXcf.cpp \\\n"
            "       ./StreAlign/CStretchCC2D.cpp \\\n"
            "       ./StreAlign/CStretchAlign.cpp \\\n"
            "       ./StreAlign/CStreAlignMain.cpp \\\n"
            "       ./CommonLine/CCommonLineMain.cpp \\\n"
            "       ./CommonLine/CFindTiltAxis.cpp \\\n"
            "       ./CommonLine/CRefineTiltAxis.cpp \\\n"
            "       ./CommonLine/CGenLines.cpp \\\n"
            "       ./CommonLine/CCalcScore.cpp \\\n"
            "       ./CommonLine/CPossibleLines.cpp \\\n"
            "       ./CommonLine/CLineSet.cpp \\\n"
            "       ./CommonLine/CSumLines.cpp \\\n"
            "       ./CommonLine/CCommonLineParam.cpp \\\n"
            "       ./Massnorm/CLinearNorm.cpp \\\n"
            "       ./Massnorm/CPositivity.cpp \\\n"
            "       ./Massnorm/CFlipInt3D.cpp \\\n"
            "       ./Correct/CCorrectUtil.cpp \\\n"
            "       ./Correct/CBinStack.cpp \\\n"
            "       ./Correct/CCorrProj.cpp \\\n"
            "       ./Correct/CCorrTomoStack.cpp \\\n"
            "       ./Correct/CFourierCropImage.cpp \\\n"
            "       ./Correct/CCorrLinearInterp.cpp \\\n"
            "       ./DoseWeight/CWeightTomoStack.cpp \\\n"
            "       ./Recon/CTomoWbp.cpp \\\n"
            "       ./Recon/CTomoSart.cpp \\\n"
            "       ./Recon/CDoBaseRecon.cpp \\\n"
            "       ./Recon/CDoSartRecon.cpp \\\n"
            "       ./Recon/CDoWbpRecon.cpp \\\n"
            "       ./TiltOffset/CTiltOffsetMain.cpp \\\n"
            "       ./PatchAlign/CFitPatchShifts.cpp \\\n"
            "       ./PatchAlign/CExtTomoStack.cpp \\\n"
            "       ./PatchAlign/CLocalAlign.cpp \\\n"
            "       ./PatchAlign/CDetectFeatures.cpp \\\n"
            "       ./PatchAlign/CRoiTargets.cpp \\\n"
            "       ./PatchAlign/CPatchTargets.cpp \\\n"
            "       ./PatchAlign/CPatchAlignMain.cpp \\\n"
            "       ./CInput.cpp \\\n"
            "       ./CFFTBuffer.cpp \\\n"
            "       ./CProcessThread.cpp \\\n"
            "       ./CAreTomoMain.cpp \\\n"
            "       $(CUCPPS)\n"
            "OBJS = $(patsubst %.cpp, %.o, $(SRCS))\n"
            "\n"
            "#-------------------------------------\n"
            "# Host and CUDA compilers.\n"
            "# Using C++14 to ensure better compatibility on modern systems.\n"
            "CC = g++ -std=c++14\n"
            "\n"
            "# Added -fPIC to generate Position Independent Code\n"
            "CFLAG = -c -g -pthread -m64 -fPIC\n"
            "\n"
            "# Added --allow-unsupported-compiler to prevent NVCC from failing on Ubuntu 24.04 with modern GCC\n"
            "NVCC = $(CUDAHOME)/bin/nvcc -std=c++14 --allow-unsupported-compiler\n"
            "\n"
            "# NVCC flags. \n"
            "# -arch=all-major automatically generates the Fat Binary without manually specifying architectures.\n"
            "# -Xcompiler -fPIC passes the position-independent flag to the host compiler.\n"
            "CUFLAG = -Xptxas -dlcm=ca -O2 -arch=all-major -Xcompiler -fPIC\n"
            "\n"
            "#------------------------------------------\n"
            "cuda: $(CUCPPS)\n"
            "\n"
            "compile: $(OBJS)\n"
            "\n"
            "# Added -no-pie to the linker flags. This bypasses the PIE requirement and \n"
            "# fixes the error caused by libmrcfile.a being compiled without -fPIC.\n"
            "exe: $(OBJS)\n"
            "\t@g++ -g -pthread -m64 -no-pie $(OBJS) \\\n"
            "\t$(PRJLIB)/libmrcfile.a \\\n"
            "\t$(PRJLIB)/libutil.a \\\n"
            "\t-L$(CUDALIB) -L/usr/lib/x86_64-linux-gnu -L/usr/lib64 \\\n"
            "\t-lcufft -lcudart -lcuda -lc -lm -lpthread \\\n"
            "\t-o AreTomo2\n"
            "\t@echo AreTomo2 has been generated.\n"
            "\n"
            "%.o: %.cu\n"
            "\t@$(NVCC) -c $(CUFLAG) -I$(PRJINC) -I$(CUDAINC) $< -o $@\n"
            "\t@echo $< has been compiled.\n"
            "\n"
            "%.o: %.cpp\n"
            "\t@$(CC) $(CFLAG) -I$(PRJINC) -I$(CUDAINC) \\\n"
            "\t\t$< -o $@\n"
            "\t@echo $< has been compiled.\n"
            "\n"
            "clean:\n"
            "\t@rm -f $(OBJS) $(CUCPPS) *.h~ makefile~ AreTomo2\n"
        )

        # Write the file
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(contents)
                makefilePath = f.name
                return makefilePath
        except Exception as e:
            raise e
