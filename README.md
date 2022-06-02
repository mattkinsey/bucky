# Bucky Model 
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://bucky.readthedocs.io/en/latest/)
![black-flake8-isort-hooks](https://github.com/mattkinsey/bucky/workflows/black-flake8-isort-hooks/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/mattkinsey/bucky/badge/master)](https://www.codefactor.io/repository/github/mattkinsey/bucky/overview/master)
![Interrogate](docs/_static/interrogate_badge.svg)

**[Documentation](https://bucky.readthedocs.io/en/latest/)**

**[Developer Guide](https://github.com/mattkinsey/bucky/blob/poetry/dev_readme.md)**

The Bucky model is a spatial SEIR model for simulating COVID-19 at the county level. 

## Getting Started

### Requirements
The Bucky model currently supports Linux and OSX and includes GPU support for accelerated modeling and processing.

* ``git`` must be installed and in your PATH.
* GPU support requires a cupy-compatible CUDA installation. See the [CuPy docs](https://docs.cupy.dev/en/stable/install.html#requirements) for details.

### Installation

| !! Until the PyPi release you'll need to install via poetry !!                                            |
|:----------------------------------------------------------------------------------------------------------|
| See the [Developer Guide](https://github.com/mattkinsey/bucky/blob/poetry/dev_readme.md) for instructions.|

Standard installation:
```bash
pip install bucky-covid
```
GPU installation:
```bash
pip install bucky-covid[gpu]
```

### Configuration (TODO this is WIP)
To use a customized configuration you first need to make a local copy of the bucky configuration. In your working directory:
```bash
bucky cfg install-local
```

### Download Input Data
To download the required input data to the ``data_dir`` specified in the configuration files:
```bash
bucky data sync
```

### Running the Model

```bash
bucky run model [TODO enum params]
bucky run postprocess
```


---------------------

**!!! EVERYTHING BELOW HERE IS OUTDATED FOR THE NEW CLI !!!**

In order to illustrate how to run the model, this section contains the commands needed to run a small simulation. First, create the intermediate graph format used by the model. This graph contains county-level data on the nodes and mobility information on the edges. The command below creates a US graph for a simulation that will start on October 1, 2020. 

```console
./bmodel make_input_graph -d 2020-10-01
```

After creating the graph, run the model with 100 iterations and 20 days:

```console
./bmodel model -n 100 -d 20
```

This will create a folder in the `raw_output` directory with the unique run ID. The script `postprocess` processes and aggregates the Monte Carlo runs. This script by default postprocesses the most recent data in the `raw_output` directory and aggregates at the national, state, and county level.

```console
./bmodel postprocess
```

### Visualizing Results
To create plots:

```console
./bmodel viz.plot
```

Like postprocessing, this script by default creates plots for the most recently processed data. Plots will be located in `output/<run_id>/plots`. These plots can be customized to show different columns and historical data. See the documentation for more.

### Lookup Tables
During postprocessing, the graph file is used to define geographic relationships between administrative levels (e.g. counties, states). In some cases, a user may want to define custom geographic groupings for visualization and analysis. For example, the National Capital Region includes counties from Maryland and Virginia along with Washington, DC. An example lookup table for this region (also known as the DMV) is included in the repo, *DMV.lookup*. 

To aggregate data with this lookup table, use the flag `--lookup` followed by the path to the lookup file:

```console
    ./bmodel postprocess --lookup DMV.lookup
```
This will create a new directory with the prefix *DMV_* in the default output directory (output/DMV_<run_id>/). To plot:

```console
  ./bmodel model viz.plot --lookup DMV.lookup
```
