Getting Started
+++++++++++++++

This should work on Linux or OSX.

Clone repo

.. code-block:: bash

  git clone https://gitlab.com/kinsemc/bucky.git

Create Anaconda env (use GPU is you have an nvidia gpu that you want to use)

.. code-block:: bash

  conda env create --file environment[_gpu].yml
  conda activate bucky[_gpu]

(Optional) Edit config.yml to change the location of data and output directories

.. code-block:: bash

  vim config.yml

Download all the data

.. code-block:: bash

  ./get_US_data.sh

Create the intermediate graph format used by the model starting on October 1:

.. code-block:: bash

  ./bmodel make_input_graph -d 2020-10-01

Run the model with 100 iterations

.. code-block:: bash

  ./bmodel model -n 100

Postprocess and aggregate the monte carlo runs

.. code-block:: bash

  ./bmodel postprocess

Create output plots (they will be in output/<run_id>/plots)

.. code-block:: bash

  ./bmodel viz.plot

During postprocessing, the graph file is used to define geographic relationships between administrative levels (e.g. counties, states). In some cases, a user may want to define custom geographic groupings for visualization and analysis. For example, the National Capital Region includes counties from Maryland and Virginia along with Washington, DC. An example lookup table for this region (also known as the DMV) is included in the repo, *DMV.lookup*. 

To aggregate data with this lookup table, use the flag ``--lookup`` followed by the path to the lookup file:

.. code-block:: bash

    ./bmodel postprocess --lookup DMV.lookup

This will create a new directory with the prefix *DMV_* in the default output directory (output/DMV_<run_id>/). To plot:

.. code-block:: bash

  ./bmodel model viz.plot --lookup DMV.lookup
