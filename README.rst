==============
AreTomo plugin
==============

This plugin provides a wrapper for `AreTomo <https://msg.ucsf.edu/software>`_ program.

.. image:: https://img.shields.io/pypi/v/scipion-em-aretomo.svg
        :target: https://pypi.python.org/pypi/scipion-em-aretomo
        :alt: PyPI release

.. image:: https://img.shields.io/pypi/l/scipion-em-aretomo.svg
        :target: https://pypi.python.org/pypi/scipion-em-aretomo
        :alt: License

.. image:: https://img.shields.io/pypi/pyversions/scipion-em-aretomo.svg
        :target: https://pypi.python.org/pypi/scipion-em-aretomo
        :alt: Supported Python versions

.. image:: https://img.shields.io/sonar/quality_gate/scipion-em_scipion-em-aretomo?server=https%3A%2F%2Fsonarcloud.io
        :target: https://sonarcloud.io/dashboard?id=scipion-em_scipion-em-aretomo
        :alt: SonarCloud quality gate

.. image:: https://img.shields.io/pypi/dm/scipion-em-aretomo
        :target: https://pypi.python.org/pypi/scipion-em-aretomo
        :alt: Downloads

Installation
------------

You will need to use 3.0+ version of Scipion to be able to run these protocols. To install the plugin, you have two options:

a) Stable version

.. code-block::

    scipion installp -p scipion-em-aretomo

b) Developer's version

    * download repository

    .. code-block::

        git clone -b devel https://github.com/scipion-em/scipion-em-aretomo.git

    * install

    .. code-block::

        scipion installp -p /path/to/scipion-em-aretomo --devel

* AreTomo binaries will be installed automatically with the plugin, but you can also link an existing installation.
* Default installation path assumed is ``software/em/aretomo-1.3.4``, if you want to change it, set *ARETOMO_HOME* in ``scipion.conf`` file to the folder where the AreTomo is installed.
* Depending on your CUDA version this plugin will guess the right default binary from AreTomo_1.3.4_CudaXY_Feb22_2023 (X is for cuda major version, Y for the minor). You can always set a different one by explicitly setting ARETOMO_BIN variable.
* If you need to use CUDA different from the one used during Scipion installation (defined by CUDA_LIB), you can add *ARETOMO_CUDA_LIB* variable to the config file. Various binaries can be downloaded from the official UCSF website.

To check the installation, simply run the following Scipion test:

``scipion test aretomo.tests.test_protocols_aretomo.TestAreTomo``

Licensing
---------

AreTomo is free for academic use only. For commercial use, please contact David Agard or Yifan Cheng for licensing:

    * agard@msg.ucsf.edu
    * Yifan.Cheng@ucsf.edu

Supported versions
------------------

1.1.0, 1.1.1, 1.2.0, 1.2.5, 1.3.0, 1.3.3, 1.3.4

Protocols
---------

    * tilt-series align and reconstruct

Detailed manual can be found in ``software/em/aretomo-1.3.4/bin/AreTomoManual_1.3.0_09292022.pdf``

References
----------

    * AreTomo: An integrated software package for automated marker-free, motion-corrected cryo-electron tomographic alignment and reconstruction. Shawn Zheng, Georg Wolff, Garrett Greenan, Zhen Chen, Frank G. A. Faas, Montserrat Bárcena, Abraham J. Koster, Yifan Cheng, David Agard. JSB vol. 6, 2022, 100068.
