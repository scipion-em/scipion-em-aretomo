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

import os

import pwem
import pyworkflow.utils as pwutils

from .constants import *


__version__ = '3.8.2'
_logo = "aretomo_logo.png"
_references = ['Zheng2022']


class Plugin(pwem.Plugin):
    _homeVar = ARETOMO_HOME
    _pathVars = [ARETOMO_HOME, ARETOMO_CUDA_LIB]
    _supportedVersions = [V1_0_0, V1_3_4]
    _url = "https://github.com/scipion-em/scipion-em-aretomo"

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(ARETOMO_HOME, f'aretomo2-{V1_0_0}')
        cls._defineVar(ARETOMO_CUDA_LIB, pwem.Config.CUDA_LIB)

        # Define the variable default value based on the guessed cuda version
        cudaVersion = cls.guessCudaVersion(ARETOMO_CUDA_LIB)
        cls._defineVar(ARETOMO_BIN,
                       f'AreTomo2_{V1_0_0}_Cuda{cudaVersion.major}{cudaVersion.minor}')

    @classmethod
    def getEnviron(cls):
        """ Return the environment to run AreTomo. """
        environ = pwutils.Environ(os.environ)
        # Get AreTomo CUDA library path if defined
        cudaLib = cls.getVar(ARETOMO_CUDA_LIB, pwem.Config.CUDA_LIB)
        environ.addLibrary(cudaLib)

        return environ

    @classmethod
    def versionGE(cls, version: str) -> bool:
        """ Return True if current version of AreTomo is newer
         or equal than the input argument.
         Params:
            version: string version (semantic version, e.g 1.0.12)
        """
        v1 = cls.getActiveVersion()
        if v1 not in cls._supportedVersions:
            raise ValueError("This version of AreTomo is not supported: ", v1)

        if cls._supportedVersions.index(v1) < cls._supportedVersions.index(version):
            return False
        return True

    @classmethod
    def getProgram(cls) -> str:
        """ Return the program binary that will be used. """
        return cls.getHome('bin', cls.getVar(ARETOMO_BIN))

    @classmethod
    def defineBinaries(cls, env):
        env.addPackage('aretomo2', version=V1_0_0,
                       tar=f"aretomo2-{V1_0_0}.tgz",
                       default=True)
        env.addPackage('aretomo', version=V1_3_4,
                       tar=f"aretomo_v{V1_3_4}.tgz")
