
# Getting Started

### Requirements
The Bucky model currently supports Linux and OSX and includes GPU support for accelerated modeling and processing. Anaconda environment files are provided for installation of dependencies. 

### Installation
Clone the repo using git:

```console
git clone https://github.com/OCHA-DAP/pa-ocha-bucky.git
```

Create Anaconda environment using `environment.yml` or `environment_gpu.yml` if using the GPU.

```console
conda env create --file environment[_gpu].yml
conda activate bucky[_gpu]
```

*Optional*: Data and output directory default locations are defined in `config.yml`. Edit this file to change these.

Add an input graph file to the repository. An example file for Afghanistan is given in *included_data/graphs/AFG/AFG_graph_20201014.p*



### Running the Model
In order to illustrate how to run the model, this section contains the commands needed to run a small simulation. 
First, add an intermediate graph format used by the model to the directory. An example file for Afghanistan is given in *included_data/graphs/AFG/AFG_graph_20201014.p* 
This graph contains admin2-level data on the nodes and mobility information on the edges. 

After adding the graph in pickle format, create a .csv file with the historical data from the graph

```console
    ./bmodel util.graph2histcsv included_data/graphs/AFG_graph_20201014.p included_data/graphs/AFG_hist_20201014.csv
```

Now you can run the model. An exmaple with 100 iterations and 20 days:

```console
./bmodel model -n 100 -d 20 -g included_data/graphs/AFG_graph_20201014.p
```

This will create a folder in the `raw_output` directory with the unique run ID. 

In the above example, no Non-Pharmeutical Interventions (NPIs) are defined and thus the model bases its values on historical data without explicitly taking into account any NPIs nor explicitly lifting them.
If it is desirable to either model the NPIs or explicitly lift them, a csv with the currently inplace NPIs should be added. An example NPI file for Afghanistan is given in *included_data/npi/AFG_NPIs_20201014.csv*. The simulation with NPIs can be run by

```console
  ./bmodel model -n 100 -d 20 -g included_data/graphs/AFG_graph_20201014.p --npi_file included_data/npi/AFG_NPIs_20201014.csv
```

Lastly, there is the option to explicitly lift the NPIs. This can be done by adding the ``--disable-npi`` argument, i.e.

```console
  ./bmodel model -n 100 -d 20 -g included_data/graphs/AFG_graph_20201014.p --npi_file included_data/npi/AFG_NPIs_20201014.csv --disable-npi
```

The script `postprocess` processes and aggregates the Monte Carlo runs. 
This script by default postprocesses the most recent data in the `raw_output` directory and aggregates at the national, admin1, and admin2 level.

```console
./bmodel postprocess -g included_data/graphs/AFG_graph_20201014.p
```

### Visualizing Results
To create plots:

```console
./bmodel viz.plot -g included_data/graphs/AFG_graph_20201014.p
```

Like postprocessing, this script by default creates plots for the most recently processed data. Plots will be located in `output/<run_id>/plots`. These plots can be customized to show different columns and historical data. See the documentation for more.

### Lookup Tables
During postprocessing, the graph file is used to define geographic relationships between administrative levels. In some cases, a user may want to define custom geographic groupings for visualization and analysis. 
For example, the captial region might include several admin regions. An example lookup table for te capital region of the US (also known as the DMV) is included in the repo, *DMV.lookup*. 

To aggregate data with this lookup table, use the flag `--lookup` followed by the path to the lookup file:

```console
    ./bmodel postprocess --lookup DMV.lookup
```
This will create a new directory with the prefix *DMV_* in the default output directory (output/DMV_<run_id>/). To plot:

```console
  ./bmodel model viz.plot --lookup DMV.lookup
```
