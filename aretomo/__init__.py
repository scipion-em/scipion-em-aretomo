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


__version__ = '3.0.3'
_logo = "aretomo_logo.png"
_references = ['Zheng']


class Plugin(pwem.Plugin):
    _homeVar = ARETOMO_HOME
    _pathVars = [ARETOMO_HOME]
    _supportedVersions = [V1_0_6, V1_0_8, V1_0_10, V1_0_12]
    _url = "https://github.com/scipion-em/scipion-em-aretomo"

    @classmethod
    def _defineVariables(cls):
        cls._defineEmVar(ARETOMO_HOME, 'aretomo-%s' % V1_0_12)
        cls._defineVar(ARETOMO_BIN, 'AreTomo_1.0.12_Cuda101')
        cls._defineVar(ARETOMO_CUDA_LIB, pwem.Config.CUDA_LIB)

    @classmethod
    def getEnviron(cls):
        """ Return the environment to run AreTomo. """
        environ = pwutils.Environ(os.environ)
        # Get AreTomo CUDA library path if defined
        cudaLib = cls.getVar(ARETOMO_CUDA_LIB, pwem.Config.CUDA_LIB)
        environ.addLibrary(cudaLib)

        return environ

    @classmethod
    def getProgram(cls):
        """ Return the program binary that will be used. """
        return cls.getHome('bin', cls.getVar(ARETOMO_BIN))

    @classmethod
    def defineBinaries(cls, env):
        for v in cls._supportedVersions:
            env.addPackage('aretomo', version=v,
                           tar='aretomo_v%s.tgz' % v,
                           default=v == V1_0_12)
