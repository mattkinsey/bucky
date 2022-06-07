===============
Quickstart
===============

Requirements
------------
The Bucky model currently supports Linux and includes GPU support for accelerated modeling and processing.

* GPU support is provided via `CuPy <https://cupy.dev/>`_. You'll need to ensure that your system is compatible with `CuPy's requirements <https://docs.cupy.dev/en/stable/install.html#requirements>`_, namely that you have an NVIDIA CUDA GPU and CUDA Toolkit version 10.2+ installed prior to installing bucky.

* Python version v3.8.0+ / v3.9.0+ / v3.10.0+. If your system has an older Python release we recommend installing bucky in an `anaconda <https://www.anaconda.com/>`_ environment. Instructions to install conda are availible in the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

* git must be installed on you system and in you `PATH`.

Installation
------------
.. note::
    This will install bucky for execution ONLY. If you plan to develop the model you'll need to install it via the instuctions found `here <https://github.com/mattkinsey/bucky/blob/master/dev_readme.md>`_

Install bucky via `pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

    pip install bucky-covid


Setting a working directory
---------------------------
Bucky will produce multiple folders for downloaded historical data and outputs. It's recommended to put these in their own directory, for example `~/bucky`, and excute the bucky CLI from that directory.

.. code-block:: bash

    BUCKY_DIR=~/bucky
    mkdir $BUCKY_DIR
    cd $BUCKY_DIR

.. note::
   The location of these directories can be globally specified in the configuration files. TODO link to config

Running the Model
-----------------

In order to illustrate how to run the model, this section contains the commands needed to run a small simulation. First, you have to download the input data required for a simulation:

.. code-block:: bash

    bucky data sync

.. note::
    By default this data will be save to `<pwd>/data`.

You can now run the model, calculate quantile estimates and generate some plots. For example, to run the model with 100 Monte Carlo iterations and 20 days:

.. code-block:: bash

    bucky run -n 100 -d 20

Equivalently, you can run each step on it's own:

.. code-block:: bash

    bucky run model -n 100 -d 20
    bucky run postprocess
    bucky viz plot

Running the model will produce output csvs and plots located in the specified `output_dir`, by defualt this will be located at `<pwd>/output`.
