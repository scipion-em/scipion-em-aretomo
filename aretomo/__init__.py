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
from aretomo.constants import ARETOMO_HOME, ARETOMO_CUDA_LIB, DEFAULT_VERSION, ARETOMO3, COMMIT_2026_09_21
from pyworkflow import VarTypes, TOMO


__version__ = '5.0.0'
_logo = "aretomo_logo.png"
_references = ['Zheng2022']


class Plugin(pwem.Plugin):
    _homeVar = ARETOMO_HOME
    _pathVars = [ARETOMO_HOME, ARETOMO_CUDA_LIB]
    _supportedVersions = [COMMIT_2026_09_21]
    _url = "https://github.com/czimaginginstitute/AreTomo3"
    _processingField = [TOMO]

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(ARETOMO_HOME, f'{ARETOMO3}-{DEFAULT_VERSION}',
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
        return cls.getHome(ARETOMO3)

    @classmethod
    def defineBinaries(cls, env):
        ARETOMO_INSTALLED = 'aretomo_installed'
        MAKEFILE11 = 'makefile11'
        aretomoHome = cls.getVar(ARETOMO_HOME)
        fixedMakeFile = cls.__createFatBinaryMakeFile()
        cmd = [
            f'cd .. && rm -rf {aretomoHome} && '
            f'git clone https://github.com/czimaginginstitute/AreTomo3.git {aretomoHome} && '
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

        env.addPackage(ARETOMO3, version=DEFAULT_VERSION,
                        tar = 'void.tgz',
                        neededProgs = ['git', 'gcc', 'g++', 'make', 'cmake'],
                        commands = installationCmds,
                        updateCuda = True,
                        default = True)

    @classmethod
    def __createFatBinaryMakeFile(cls) -> str:
        """
        Creates a generalized makefile11 that replaces, at installation time, the
        original AreTomo3 makefile11. Compared to the original one, it:

        - Locates the CUDA toolkit by itself: CUDAHOME, then CUDA_HOME / CUDA_PATH /
          CUDA_ROOT, nvcc on PATH, /usr/local/cuda and the newest versioned install.
        - Builds a fat binary: native code for each of sm_61...sm_120 supported by
          the detected nvcc, plus PTX for the newest (override with CUDA_ARCHS).
        - Checks up front for g++, a g++/CUDA version mismatch and the libtiff
          headers, stopping with a plain message (bypass with SKIP_CHECKS=1).
        - Compiles LibSrc/Util and LibSrc/Mrcfile from source instead of linking
          the prebuilt archives in LibSrc/Lib.
        - Embeds the CUDA lib dir as RUNPATH (disable with CUDA_RPATH=) and links
          NVTX only if the toolkit provides it.

        :return: the path of the temporary file containing the makefile.
        """
        # Define the contents of the file. The source lists (CUSRCS, SRCS) must match
        # the AreTomo3 revision that is cloned in defineBinaries.
        # NOTE: Explicit tab characters (\t) are used in the recipe commands and in the
        # continuation lines to comply with the Make standard.
        contents = (
            "PRJHOME = $(shell pwd)\n"
            "# >>> patched by tools/patch_makefile11.py (CUDA detection)\n"
            "# Search order: CUDAHOME given by the user, CUDA_HOME / CUDA_PATH /\n"
            "# CUDA_ROOT from the environment, nvcc found on PATH, /usr/local/cuda,\n"
            "# then the newest versioned install in the usual places.\n"
            "CUDA_NVCC_PATH := $(realpath $(shell command -v nvcc 2>/dev/null))\n"
            "CUDA_NVCC_DISK := $(wildcard /usr/local/cuda/bin/nvcc) \\\n"
            "\t$(shell ls -d /usr/local/cuda-*/bin/nvcc /opt/cuda/bin/nvcc \\\n"
            "\t$(HOME)/nvidia/cuda-*/bin/nvcc 2>/dev/null | sort -rV)\n"
            "CUDAHOME ?= $(firstword $(CUDA_HOME) $(CUDA_PATH) $(CUDA_ROOT) \\\n"
            "\t$(patsubst %/bin/nvcc,%,$(CUDA_NVCC_PATH) $(CUDA_NVCC_DISK)))\n"
            "ifeq ($(and $(strip $(CUDAHOME)),$(wildcard $(CUDAHOME)/bin/nvcc)),)\n"
            "ifneq ($(MAKECMDGOALS),clean)\n"
            "$(error CUDA toolkit not found (looked at \"$(CUDAHOME)\"). Install it, \\\n"
            "put nvcc on PATH, or run make with CUDAHOME=/path/to/cuda)\n"
            "endif\n"
            "endif\n"
            "CUDAINC = $(CUDAHOME)/include\n"
            "CUDALIB = $(firstword $(wildcard $(CUDAHOME)/lib64 $(CUDAHOME)/lib) \\\n"
            "\t$(CUDAHOME)/lib64)\n"
            "# NVTX was removed from recent toolkits; link it only when present.\n"
            "NVTXLIB = $(if $(wildcard $(CUDALIB)/libnvToolsExt.*),-lnvToolsExt)\n"
            "# Runtime search path, so the binary finds the CUDA libraries it was\n"
            "# built against without LD_LIBRARY_PATH (which still takes precedence).\n"
            "# Not needed for a system-wide toolkit under /usr. Disable with\n"
            "#   make exe -f makefile11 CUDA_RPATH=\n"
            "comma := ,\n"
            "CUDA_RPATH ?= $(if $(filter /usr,$(CUDAHOME)),,$(CUDALIB))\n"
            "RPATHFLAG = $(if $(strip $(CUDA_RPATH)),-Xlinker -rpath$(comma)$(CUDA_RPATH))\n"
            "# <<< patched\n"
            "PRJINC = $(PRJHOME)/LibSrc/Include\n"
            "PRJLIB = $(PRJHOME)/LibSrc/Lib\n"
            "#-----------------------------\n"
            "CUSRCS = ./MaUtil/GAddFrames.cu \\\n"
            "\t./MaUtil/GCalcFRC.cu \\\n"
            "\t./MaUtil/GCalcMoment2D.cu \\\n"
            "\t./MaUtil/GCorrLinearInterp.cu \\\n"
            "\t./MaUtil/GFFT1D.cu \\\n"
            "\t./MaUtil/GFFTUtil2D.cu \\\n"
            "\t./MaUtil/GFillEmpty2D.cu \\\n"
            "\t./MaUtil/GFindMinMax2D.cu \\\n"
            "\t./MaUtil/GFourierResize2D.cu \\\n"
            "\t./MaUtil/GFtResize2D.cu \\\n"
            "\t./MaUtil/GGenRandoms.cu \\\n"
            "\t./MaUtil/GNormalize2D.cu \\\n"
            "\t./MaUtil/GPad2D.cu \\\n"
            "\t./MaUtil/GPartialCopy.cu \\\n"
            "\t./MaUtil/GPhaseShift2D.cu \\\n"
            "\t./MaUtil/GPositivity2D.cu \\\n"
            "\t./MaUtil/GRoundEdge1D.cu \\\n"
            "\t./MaUtil/GRoundEdge2D.cu \\\n"
            "\t./MaUtil/GThreshold2D.cu \\\n"
            "\t./MotionCor/MrcUtil/G90Rotate2D.cu \\\n"
            "\t./MotionCor/MrcUtil/GApplyRefsToFrame.cu \\\n"
            "\t./MotionCor/MrcUtil/GAugmentRef.cu \\\n"
            "\t./MotionCor/MrcUtil/GFlip2D.cu \\\n"
            "\t./MotionCor/MrcUtil/GInverse2D.cu \\\n"
            "\t./MotionCor/EerUtil/GAddRawFrame.cu \\\n"
            "\t./MotionCor/BadPixel/GCombineBadMap.cu \\\n"
            "\t./MotionCor/BadPixel/GCorrectBad.cu \\\n"
            "\t./MotionCor/BadPixel/GDetectHot.cu \\\n"
            "\t./MotionCor/BadPixel/GDetectPatch.cu \\\n"
            "\t./MotionCor/BadPixel/GLabelPatch.cu \\\n"
            "\t./MotionCor/BadPixel/GLocalCC.cu \\\n"
            "\t./MotionCor/Align/GCC2D.cu \\\n"
            "\t./MotionCor/Align/GCorrelateSum2D.cu \\\n"
            "\t./MotionCor/Align/GNormByStd2D.cu \\\n"
            "\t./MotionCor/Correct/GCorrectPatchShift.cu \\\n"
            "\t./MotionCor/Correct/GStretch.cu \\\n"
            "\t./MotionCor/Correct/GWeightFrame.cu \\\n"
            "\t./MotionCor/MotionDecon/GDeconFrame.cu \\\n"
            "\t./MotionCor/MotionDecon/GMotionWeight.cu \\\n"
            "\t./AreTomo/Util/GBinImage2D.cu \\\n"
            "\t./AreTomo/Util/GCC1D.cu \\\n"
            "\t./AreTomo/Util/GCC2D.cu \\\n"
            "\t./AreTomo/Util/GLocalCC2D.cu \\\n"
            "\t./AreTomo/Util/GLocalRms2D.cu \\\n"
            "\t./AreTomo/Util/GMutualMask2D.cu \\\n"
            "\t./AreTomo/Util/GRealCC2D.cu \\\n"
            "\t./AreTomo/Util/GRemoveSpikes2D.cu \\\n"
            "\t./AreTomo/Util/GRotate2D.cu \\\n"
            "\t./AreTomo/Util/GShiftRotate2D.cu \\\n"
            "\t./AreTomo/Util/GTiltStretch.cu \\\n"
            "\t./AreTomo/Util/GXcf2D.cu \\\n"
            "\t./AreTomo/Massnorm/GFlipInt2D.cu \\\n"
            "\t./AreTomo/CommonLine/GCalcCommonRegion.cu \\\n"
            "\t./AreTomo/CommonLine/GCoherence.cu \\\n"
            "\t./AreTomo/CommonLine/GFunctions.cu \\\n"
            "\t./AreTomo/CommonLine/GGenCommonLine.cu \\\n"
            "\t./AreTomo/CommonLine/GInterpolateLineSet.cu \\\n"
            "\t./AreTomo/CommonLine/GRemoveMean.cu \\\n"
            "\t./AreTomo/CommonLine/GSumLines.cu \\\n"
            "\t./AreTomo/Correct/GCorrPatchShift.cu \\\n"
            "\t./AreTomo/Correct/GRoundProjEdge.cu \\\n"
            "\t./AreTomo/DoseWeight/GDoseWeightImage.cu \\\n"
            "\t./AreTomo/FindCtf/GBackground1D.cu \\\n"
            "\t./AreTomo/FindCtf/GCalcCTF1D.cu \\\n"
            "\t./AreTomo/FindCtf/GCalcCTF2D.cu \\\n"
            "\t./AreTomo/FindCtf/GCalcSpectrum.cu \\\n"
            "\t./AreTomo/FindCtf/GCC1D.cu \\\n"
            "\t./AreTomo/FindCtf/GCC2D.cu \\\n"
            "\t./AreTomo/FindCtf/GSpectralCC2D.cu \\\n"
            "\t./AreTomo/FindCtf/GCorrCTF2D.cu \\\n"
            "\t./AreTomo/FindCtf/GRadialAvg.cu \\\n"
            "\t./AreTomo/FindCtf/GRemoveMean.cu \\\n"
            "\t./AreTomo/FindCtf/GRmBackground2D.cu \\\n"
            "\t./AreTomo/FindCtf/GRmSpikes.cu \\\n"
            "\t./AreTomo/FindCtf/GRoundEdge.cu \\\n"
            "\t./AreTomo/FindCtf/GExtractTile.cu \\\n"
            "\t./AreTomo/FindCtf/GScaleSpect2D.cu \\\n"
            "\t./AreTomo/PatchAlign/GCommonArea.cu \\\n"
            "\t./AreTomo/PatchAlign/GExtractPatch.cu \\\n"
            "\t./AreTomo/PatchAlign/GGenXcfImage.cu \\\n"
            "\t./AreTomo/PatchAlign/GNormByStd2D.cu \\\n"
            "\t./AreTomo/ProjAlign/GProjXcf.cu \\\n"
            "\t./AreTomo/ProjAlign/GReproj.cu \\\n"
            "\t./AreTomo/Recon/GBackProj.cu \\\n"
            "\t./AreTomo/Recon/GBackProjWbp.cu \\\n"
            "\t./AreTomo/Recon/GCalcRFactor.cu \\\n"
            "\t./AreTomo/Recon/GDiffProj.cu \\\n"
            "\t./AreTomo/Recon/GForProj.cu \\\n"
            "\t./AreTomo/Recon/GRWeight.cu \\\n"
            "\t./AreTomo/Recon/GWeightProjs.cu\n"
            "CUCPPS = $(patsubst %.cu, %.cpp, $(CUSRCS))\n"
            "#------------------------------------------\n"
            "SRCS = ./MaUtil/CParseArgs.cpp \\\n"
            "\t./MaUtil/CCufft2D.cpp \\\n"
            "\t./MaUtil/CFileName.cpp \\\n"
            "\t./MaUtil/CPad2D.cpp \\\n"
            "\t./MaUtil/CPeak2D.cpp \\\n"
            "\t./MaUtil/CSaveTempMrc.cpp \\\n"
            "\t./MaUtil/CSimpleFuncs.cpp \\\n"
            "\t./DataUtil/CAlnSums.cpp \\\n"
            "\t./DataUtil/CAsyncSaveVol.cpp \\\n"
            "\t./DataUtil/CBufferPool.cpp \\\n"
            "\t./DataUtil/CCtfResults.cpp \\\n"
            "\t./DataUtil/CDuInstances.cpp \\\n"
            "\t./DataUtil/CGpuBuffer.cpp \\\n"
            "\t./DataUtil/CMcPackage.cpp \\\n"
            "\t./DataUtil/CMrcStack.cpp \\\n"
            "\t./DataUtil/CReadMdoc.cpp \\\n"
            "\t./DataUtil/CStackBuffer.cpp \\\n"
            "\t./DataUtil/CReadMdocDone.cpp \\\n"
            "\t./DataUtil/CSaveMdocDone.cpp \\\n"
            "\t./DataUtil/CStackFolder.cpp \\\n"
            "\t./DataUtil/CTiltSeries.cpp \\\n"
            "\t./DataUtil/CTsPackage.cpp \\\n"
            "\t./DataUtil/CCtfParam.cpp \\\n"
            "\t./DataUtil/CLogFiles.cpp \\\n"
            "\t./DataUtil/CTimeStamp.cpp \\\n"
            "\t./MotionCor/DataUtil/CFmGroupParam.cpp \\\n"
            "\t./MotionCor/DataUtil/CFmIntParam.cpp \\\n"
            "\t./MotionCor/DataUtil/CPatchShifts.cpp \\\n"
            "\t./MotionCor/DataUtil/CStackShift.cpp \\\n"
            "\t./MotionCor/BadPixel/CCorrectMain.cpp \\\n"
            "\t./MotionCor/BadPixel/CDetectMain.cpp \\\n"
            "\t./MotionCor/BadPixel/CLocalCCMap.cpp \\\n"
            "\t./MotionCor/BadPixel/CTemplate.cpp \\\n"
            "\t./MotionCor/Align/CAlignBase.cpp \\\n"
            "\t./MotionCor/Align/CAlignedSum.cpp \\\n"
            "\t./MotionCor/Align/CAlignMain.cpp \\\n"
            "\t./MotionCor/Align/CAlignParam.cpp \\\n"
            "\t./MotionCor/Align/CAlignStack.cpp \\\n"
            "\t./MotionCor/Align/CDetectFeatures.cpp \\\n"
            "\t./MotionCor/Align/CExtractPatch.cpp \\\n"
            "\t./MotionCor/Align/CFullAlign.cpp \\\n"
            "\t./MotionCor/Align/CGenXcfStack.cpp \\\n"
            "\t./MotionCor/Align/CIterativeAlign.cpp \\\n"
            "\t./MotionCor/Align/CMeasurePatches.cpp \\\n"
            "\t./MotionCor/Align/CPatchAlign.cpp \\\n"
            "\t./MotionCor/Align/CPatchCenters.cpp \\\n"
            "\t./MotionCor/Align/CSaveAlign.cpp \\\n"
            "\t./MotionCor/Align/CSimpleSum.cpp \\\n"
            "\t./MotionCor/Align/CTransformStack.cpp \\\n"
            "\t./MotionCor/Correct/CCorrectFullShift.cpp \\\n"
            "\t./MotionCor/Correct/CGenRealStack.cpp \\\n"
            "\t./MotionCor/MotionDecon/CInFrameMotion.cpp \\\n"
            "\t./MotionCor/MrcUtil/CApplyRefs.cpp \\\n"
            "\t./MotionCor/MrcUtil/CLoadMrcMain.cpp \\\n"
            "\t./MotionCor/MrcUtil/CSumFFTStack.cpp \\\n"
            "\t./MotionCor/TiffUtil/CLoadTiffHeader.cpp \\\n"
            "\t./MotionCor/TiffUtil/CLoadTiffImage.cpp \\\n"
            "\t./MotionCor/TiffUtil/CLoadTiffMain.cpp \\\n"
            "\t./MotionCor/Util/CGroupFrames.cpp \\\n"
            "\t./MotionCor/Util/CNextItem.cpp \\\n"
            "\t./MotionCor/Util/CRemoveSpikes1D.cpp \\\n"
            "\t./MotionCor/Util/CSplineFit1D.cpp \\\n"
            "\t./MotionCor/EerUtil/CLoadEerHeader.cpp \\\n"
            "\t./MotionCor/EerUtil/CLoadEerFrames.cpp \\\n"
            "\t./MotionCor/EerUtil/CDecodeEerFrame.cpp \\\n"
            "\t./MotionCor/EerUtil/CRenderMrcStack.cpp \\\n"
            "\t./MotionCor/EerUtil/CLoadEerMain.cpp \\\n"
            "\t./MotionCor/CLoadRefs.cpp \\\n"
            "\t./MotionCor/CMcInstances.cpp \\\n"
            "\t./MotionCor/CMotionCorMain.cpp \\\n"
            "\t./AreTomo/Util/CReadDataFile.cpp \\\n"
            "\t./AreTomo/Util/CSplitItems.cpp \\\n"
            "\t./AreTomo/Util/CStrLinkedList.cpp \\\n"
            "\t./AreTomo/MrcUtil/CAlignParam.cpp \\\n"
            "\t./AreTomo/MrcUtil/CCalcStackStats.cpp \\\n"
            "\t./AreTomo/MrcUtil/CCropVolume.cpp \\\n"
            "\t./AreTomo/MrcUtil/CDarkFrames.cpp \\\n"
            "\t./AreTomo/MrcUtil/CLocalAlignParam.cpp \\\n"
            "\t./AreTomo/MrcUtil/CPatchShifts.cpp \\\n"
            "\t./AreTomo/MrcUtil/CRemoveDarkFrames.cpp \\\n"
            "\t./AreTomo/MrcUtil/CSaveAlignFile.cpp \\\n"
            "\t./AreTomo/MrcUtil/CLoadAlignFile.cpp \\\n"
            "\t./AreTomo/MrcUtil/CSaveStack.cpp \\\n"
            "\t./AreTomo/MrcUtil/CMuInstances.cpp \\\n"
            "\t./AreTomo/CommonLine/CCalcScore.cpp \\\n"
            "\t./AreTomo/CommonLine/CCommonLineParam.cpp \\\n"
            "\t./AreTomo/CommonLine/CFindTiltAxis.cpp \\\n"
            "\t./AreTomo/CommonLine/CGenLines.cpp \\\n"
            "\t./AreTomo/CommonLine/CLineSet.cpp \\\n"
            "\t./AreTomo/CommonLine/CPossibleLines.cpp \\\n"
            "\t./AreTomo/CommonLine/CRefineTiltAxis.cpp \\\n"
            "\t./AreTomo/CommonLine/CSumLines.cpp \\\n"
            "\t./AreTomo/CommonLine/CCommonLineMain.cpp \\\n"
            "\t./AreTomo/Correct/CBinStack.cpp \\\n"
            "\t./AreTomo/Correct/CCorrectUtil.cpp \\\n"
            "\t./AreTomo/Correct/CCorrProj.cpp \\\n"
            "\t./AreTomo/Correct/CCorrTomoStack.cpp \\\n"
            "\t./AreTomo/Correct/CFourierCropImage.cpp \\\n"
            "\t./AreTomo/DoseWeight/CWeightTomoStack.cpp \\\n"
            "\t./AreTomo/FindCtf/CCtfTheory.cpp \\\n"
            "\t./AreTomo/FindCtf/CFindCtf1D.cpp \\\n"
            "\t./AreTomo/FindCtf/CFindCtf2D.cpp \\\n"
            "\t./AreTomo/FindCtf/CFindCtfBase.cpp \\\n"
            "\t./AreTomo/FindCtf/CFindCtfHelp.cpp \\\n"
            "\t./AreTomo/FindCtf/CFindCtfMain.cpp \\\n"
            "\t./AreTomo/FindCtf/CRefineCtfMain.cpp \\\n"
            "\t./AreTomo/FindCtf/CFindDefocus1D.cpp\\\n"
            "\t./AreTomo/FindCtf/CFindDefocus2D.cpp \\\n"
            "\t./AreTomo/FindCtf/CTile.cpp \\\n"
            "\t./AreTomo/FindCtf/CCoreTile.cpp \\\n"
            "\t./AreTomo/FindCtf/CTsTiles.cpp \\\n"
            "\t./AreTomo/FindCtf/CExtractTiles.cpp \\\n"
            "\t./AreTomo/FindCtf/CTiltInducedZ.cpp \\\n"
            "\t./AreTomo/FindCtf/CCorrImgCtf.cpp \\\n"
            "\t./AreTomo/FindCtf/CGenAvgSpectrum.cpp \\\n"
            "\t./AreTomo/FindCtf/CSaveCtfResults.cpp \\\n"
            "\t./AreTomo/FindCtf/CLoadCtfResults.cpp \\\n"
            "\t./AreTomo/FindCtf/CAlignCtfResults.cpp \\\n"
            "\t./AreTomo/FindCtf/CSpectrumImage.cpp \\\n"
            "\t./AreTomo/FindCtf/CCorrCtfMain.cpp \\\n"
            "\t./AreTomo/ImodUtil/CImodUtil.cpp \\\n"
            "\t./AreTomo/ImodUtil/CSaveCsv.cpp \\\n"
            "\t./AreTomo/ImodUtil/CSaveTilts.cpp \\\n"
            "\t./AreTomo/ImodUtil/CSaveXF.cpp \\\n"
            "\t./AreTomo/ImodUtil/CSaveXtilts.cpp \\\n"
            "\t./AreTomo/Massnorm/CFlipInt3D.cpp \\\n"
            "\t./AreTomo/Massnorm/CLinearNorm.cpp \\\n"
            "\t./AreTomo/Massnorm/CPositivity.cpp \\\n"
            "\t./AreTomo/ProjAlign/CCalcReproj.cpp \\\n"
            "\t./AreTomo/ProjAlign/CCentralXcf.cpp \\\n"
            "\t./AreTomo/ProjAlign/CParam.cpp \\\n"
            "\t./AreTomo/ProjAlign/CRemoveSpikes.cpp \\\n"
            "\t./AreTomo/ProjAlign/CProjAlignMain.cpp \\\n"
            "\t./AreTomo/PatchAlign/CDetectFeatures.cpp \\\n"
            "\t./AreTomo/PatchAlign/CFitPatchShifts.cpp \\\n"
            "\t./AreTomo/PatchAlign/CLocalAlign.cpp \\\n"
            "\t./AreTomo/PatchAlign/CPatchTargets.cpp \\\n"
            "\t./AreTomo/PatchAlign/CPatchAlignMain.cpp \\\n"
            "\t./AreTomo/Recon/CDoBaseRecon.cpp \\\n"
            "\t./AreTomo/Recon/CDoSartRecon.cpp \\\n"
            "\t./AreTomo/Recon/CDoWbpRecon.cpp \\\n"
            "\t./AreTomo/Recon/CTomoBase.cpp \\\n"
            "\t./AreTomo/Recon/CTomoSart.cpp \\\n"
            "\t./AreTomo/Recon/CTomoWbp.cpp \\\n"
            "\t./AreTomo/Recon/CCalcVolThick.cpp \\\n"
            "\t./AreTomo/Recon/CAlignMetric.cpp \\\n"
            "\t./AreTomo/StreAlign/CStretchAlign.cpp \\\n"
            "\t./AreTomo/StreAlign/CStretchCC2D.cpp \\\n"
            "\t./AreTomo/StreAlign/CStretchXcf.cpp \\\n"
            "\t./AreTomo/StreAlign/CStreAlignMain.cpp \\\n"
            "\t./AreTomo/TiltOffset/CTiltOffsetMain.cpp \\\n"
            "\t./AreTomo/CAtInstances.cpp \\\n"
            "\t./AreTomo/CTsMetrics.cpp \\\n"
            "\t./AreTomo/CAreTomoMain.cpp \\\n"
            "\t./CInput.cpp \\\n"
            "\t./CMcInput.cpp \\\n"
            "\t./CAtInput.cpp \\\n"
            "\t./CAreTomo3Json.cpp \\\n"
            "\t./CProcessThread.cpp \\\n"
            "\t./CMcAreTomoMain.cpp \\\n"
            "\t./CAreTomo3.cpp \\\n"
            "\t$(CUCPPS)\n"
            "# >>> patched by tools/patch_makefile11.py (bundled libraries)\n"
            "# Util and Mrcfile are compiled from source with everything else, so\n"
            "# the build does not depend on the prebuilt archives in LibSrc/Lib.\n"
            "# CFindMrc.cpp is not part of libmrcfile (see LibSrc/Mrcfile/makefile).\n"
            "SRCS += $(wildcard ./LibSrc/Util/*.cpp) \\\n"
            "\t$(filter-out %/CFindMrc.cpp,$(wildcard ./LibSrc/Mrcfile/*/*.cpp))\n"
            "# <<< patched\n"
            "OBJS = $(patsubst %.cpp, %.o, $(SRCS))\n"
            "#-------------------------------------\n"
            "CC = g++ -std=c++11\n"
            "CFLAG = -c -g -pthread -m64\n"
            "NVCC = $(CUDAHOME)/bin/nvcc -std=c++11\n"
            "# >>> patched by tools/patch_makefile11.py (fat binary)\n"
            "# Native code for every wanted compute capability the detected nvcc\n"
            "# supports, plus PTX of the newest so that later GPUs can still run it.\n"
            "# Override with e.g.  make exe -f makefile11 CUDA_ARCHS=\"80 86\"\n"
            "CUDA_WANTED_ARCHS = 61 70 75 80 86 89 90 100 120\n"
            "CUDA_NVCC_ARCHS := $(patsubst compute_%,%,$(shell \\\n"
            "\t$(CUDAHOME)/bin/nvcc --list-gpu-arch 2>/dev/null))\n"
            "ifeq ($(strip $(CUDA_NVCC_ARCHS)),)\n"
            "CUDA_ARCHS ?= 61 70 75 80 86 89 90\n"
            "else\n"
            "CUDA_ARCHS ?= $(filter $(CUDA_WANTED_ARCHS),$(CUDA_NVCC_ARCHS))\n"
            "endif\n"
            "CUFLAG = -Xptxas -dlcm=ca -O2 \\\n"
            "\t$(foreach a,$(CUDA_ARCHS),-gencode arch=compute_$(a)$(comma)code=sm_$(a)) \\\n"
            "\t$(foreach a,$(lastword $(CUDA_ARCHS)),-gencode arch=compute_$(a)$(comma)code=compute_$(a))\n"
            "#\n"
            "# Up-front checks, so that problems are reported in plain words before\n"
            "# anything is compiled. Skip them with  make ... SKIP_CHECKS=1\n"
            "ifneq ($(MAKECMDGOALS),clean)\n"
            "ifndef SKIP_CHECKS\n"
            "HOSTCXX := $(firstword $(CC))\n"
            "ifeq ($(shell command -v $(HOSTCXX) 2>/dev/null),)\n"
            "$(error C++ compiler \"$(HOSTCXX)\" not found. Install g++ \\\n"
            "(Debian/Ubuntu: build-essential, RHEL/Fedora: gcc-c++))\n"
            "endif\n"
            "# Each CUDA release accepts host compilers only up to some version.\n"
            "# Ask nvcc itself, since some packaged toolkits bring their own g++.\n"
            "ifneq ($(shell $(NVCC) -x cu -E /dev/null 2>&1 | \\\n"
            "\tgrep -c 'unsupported GNU version'),0)\n"
            "$(error $(HOSTCXX) $(shell $(HOSTCXX) -dumpfullversion -dumpversion) is \\\n"
            "too new for the CUDA toolkit in $(CUDAHOME) \\\n"
            "($(shell $(CUDAHOME)/bin/nvcc --version | grep -o 'release [0-9.]*')). \\\n"
            "Use a newer toolkit with CUDAHOME=/path/to/cuda or put an older g++ \\\n"
            "first on PATH. Toolkits found here: \\\n"
            "$(patsubst %/bin/nvcc,%,$(CUDA_NVCC_PATH) $(CUDA_NVCC_DISK)))\n"
            "endif\n"
            "ifneq ($(shell printf '\\043include <tiffio.h>\\n' | \\\n"
            "\t$(CC) -x c++ -fsyntax-only - >/dev/null 2>&1 || echo missing),)\n"
            "$(error libtiff headers not found. Install them \\\n"
            "(Debian/Ubuntu: libtiff-dev, RHEL/Fedora: libtiff-devel))\n"
            "endif\n"
            "endif\n"
            "endif\n"
            "# <<< patched\n"
            "#------------------------------------------\n"
            "cuda: $(CUCPPS)\n"
            "\n"
            "compile: $(OBJS)\n"
            "\n"
            "exe: $(OBJS)\n"
            "\t@$(NVCC) -g -G -m64 $(OBJS) \\\n"
            "\t-L$(CUDALIB) -L$(CUDALIB)/stubs $(RPATHFLAG) \\\n"
            "\t-L/usr/lib64 \\\n"
            "\t-lcufft -lcudart -lcuda $(NVTXLIB) -ltiff -lc -lm -lpthread \\\n"
            "\t-o AreTomo3\n"
            "\t@echo AreTomo3 has been generated.\n"
            "\n"
            "%.cpp: %.cu\n"
            "\t@echo \"-----------------------------------------------\"\n"
            "\t@$(NVCC) -cuda -cudart shared \\\n"
            "\t\t$(CUFLAG) -I$(PRJINC) \\\n"
            "\t\t$< -o $@\n"
            "\t@echo $< has been compiled.\n"
            "\n"
            "%.o: %.cpp\n"
            "\t@echo \"------------------------------------------------\"\n"
            "\t@$(CC) $(CFLAG) -I$(PRJINC) -I$(CUDAINC) \\\n"
            "\t\t$< -o $@\n"
            "\t@echo $< has been compiled.\n"
            "\n"
            "clean:\n"
            "\t@rm -f $(OBJS) $(CUCPPS) *.h~ makefile~ AreTomo3\n"
        )

        # Write the file
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(contents)
                makefilePath = f.name
                return makefilePath
        except Exception as e:
            raise e
