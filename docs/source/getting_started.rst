Getting Started
+++++++++++++++

Clone repo

.. code-block:: bash

  git clone https://gitlab.com/kinsemc/bucky.git

Create Anaconda env (use GPU is you have an nvidia gpu that you want to use)

.. code-block:: bash

  conda env create --file enviroment[_gpu].yml
  conda activate bucky[_gpu]

(Optional) Edit config.yml to change the location of data and output directories

.. code-block:: bash

  vim config.yml

Download all the data

.. code-block:: bash

  ./get_US_data.sh

Create the intermediate graph format used by the model

.. code-block:: bash

  python -m bucky.make_input_graph

Run the model with 100 iterations

.. code-block:: bash

  python -m bucky.model -n 100

Postprocess and aggregate the monte carlo runs

.. code-block:: bash

  python -m bucky.postprocess

Create output plots (they will be in output/<run_id>/plots)

.. code-block:: bash

  python -m bucky.viz.plot

TODO add DMV lookup example
