==================
Installation Guide
==================

1. To begin, first checkout the code from `GitLab <https://gitlab.com/kinsemc/bucky>`_:

.. code-block:: bash

  git clone https://gitlab.com/kinsemc/bucky.git

2. Next set up the enviroment required to run the model, first making sure `Anaconda <https://www.anaconda.com/>`_ is installed.

.. note:: Anaconda can be downloaded from `<https://docs.anaconda.com/anaconda/install/>`_

Included in the repository are two yaml formatted `Anaconda <https://www.anaconda.com/>`_ enviroment specs:

* **enviroment.yml**: Contains the standard packages required to run the model
* **enviroment_gpu.yml**: Standard enviroment + CUDA/CuPy for GPU acceleration. CuPy will be used to replace all references to numpy in the model itself.

.. note:: CuPy requires an NVIDIA GPU and will only increase performance for model runs over large geographic area (e.g. the whole US)

To install and activate the appropriate enviroment:

.. code-block:: bash

   cd bucky
   conda env create --file enviroment.yml
   conda activate bucky

or 

.. code-block:: bash

   cd bucky
   conda env create --file enviroment_gpu.yml
   conda activate bucky_gpu

3. Finally, if you wish to use custom paths to store the data associated with the model (either inputs or outputs), simply edit the contents of config.yml in the root of the repository

.. note:: It is recommended to use high speed storage for <raw_output_dir> if possible as that will have an impact on runtimes.

Downloading Input Datasets
==========================

The model depends on a number of input datasets being available in the <data_dir> specified in config.yml. To automatically download them just using the `get_US_data.sh` script provided in the root of the repository (this will take some time for the initial download):

.. code-block:: bash

   chmod +x ./get_US_data.sh
   ./get_US_data.sh

The following datasets will be automatically downloaded:

* COVID-19 Data Repository by the Center for Systems Science and Engineering at Johns Hopkins University
    * COVID-19 Case and death data on the county level
    * `GitHub <https://github.com/CSSEGISandData/COVID-19)>`_
* Descartes Labs: Data for Mobility Changes in Response to COVID-19
    * State and county-level mobility statistics
    * `GitHub <https://github.com/descarteslabs/DL-COVID-19>`_
* COVID Exposure Indices from PlaceIQ movement data
    * State and county-level location exposure indices
    * Reference: *Measuring movement and social contact with smartphone data: a real-time application to COVID-19* by Couture, Dingel, Green, Handbury, and Williams `Link <https://github.com/COVIDExposureIndices/COVIDExposureIndices/blob/master/CDGHW.pdf>`_
    * `GitHub <https://github.com/COVIDExposureIndices/COVIDExposureIndices>`_
* The COVID Tracking Project at The Atlantic
    * COVID-19 case and death data at the state level
    * `GitHub <https://github.com/COVID19Tracking/covid-tracking-data>`_
* US TIGER shapefiles from the US Census 
    * `Link <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html>`_
* US Census Bridged-Race Population estimates
    * `Link <https://www.cdc.gov/nchs/nvss/bridged_race/Documentation-Bridged-PostcenV2018.pdf>`_
* Social Contact Matrices for 152 Countries
    * *Projecting social contact matrices in 152 countries using contact surveys and demographic data*, Prem et al.
    * `Paper <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005697>`_
* USAFacts Coronavirus Stats and Data
    * County-level coronavirus cases and deaths
    * `Link <https://usafacts.org/issues/coronavirus/>`_

