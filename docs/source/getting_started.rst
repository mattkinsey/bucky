Getting Started
+++++++++++++++

This should work on Linux or OSX.

Clone repo

.. code-block:: bash

  git clone git@github.com:OCHA-DAP/pa-ocha-bucky.git

Create Anaconda env (use GPU is you have an nvidia gpu that you want to use)

.. code-block:: bash

  conda env create --file environment[_gpu].yml
  conda activate bucky[_gpu]

(Optional) Edit config.yml to change the location of data and output directories

.. code-block:: bash

  vim config.yml

Add an input graph file to the repository. An example file for Afghanistan is given in *included_data/graphs/AFG/AFG_graph_20201014.p*

Create a .csv file with the historical data from the graph

.. code-block:: bash

    ./bmodel util.graph2histcsv included_data/graphs/AFG_graph_20201014.p included_data/graphs/AFG_hist_20201014.csv

If including NPI's in the simulation, add a csv with the currently inplace NPI's. An example file for Afghanistan is given in *included_data/npi/AFG_NPIs_20201014.csv*

Run the model with 100 iterations and 20 days:

.. code-block:: bash

  ./bmodel model -n 100 -d 20 -g included_data/graphs/AFG_graph_20201014.p --npi_file included_data/npi/AFG_NPIs_20201014.csv

Postprocess and aggregate the monte carlo runs to admin 0 and admin 1 level

.. code-block:: bash

  ./bmodel postprocess -l adm0 adm1 -g included_data/graphs/AFG_graph_20201014.p

Create output plots with the projected values of all iterations (they will be in output/<run_id>/plots)

.. code-block:: bash

  ./bmodel viz.plot -l adm0 adm1 -g included_data/graphs/AFG_graph_20201014.p

During postprocessing, the graph file is used to define geographic relationships between administrative levels. In some cases, a user may want to define custom geographic groupings for visualization and analysis.
For example, the captial region might include several admin regions. An example lookup table for the capital region of the US (also known as the DMV) is included in the repo, *DMV.lookup*.

To aggregate data with this lookup table, use the flag ``--lookup`` followed by the path to the lookup file:

.. code-block:: bash

    ./bmodel postprocess --lookup DMV.lookup

This will create a new directory with the prefix *DMV_* in the default output directory (output/DMV_<run_id>/). To plot:

.. code-block:: bash

  ./bmodel model viz.plot --lookup DMV.lookup
