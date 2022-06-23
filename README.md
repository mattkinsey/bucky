# Bucky Model 
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://bucky.readthedocs.io/en/latest/)
![black-flake8-isort-hooks](https://github.com/mattkinsey/bucky/workflows/black-flake8-isort-hooks/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/mattkinsey/bucky/badge/master)](https://www.codefactor.io/repository/github/mattkinsey/bucky/overview/master)
![Interrogate](docs/_static/interrogate_badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/mattkinsey/bucky/badge.svg?branch=master)](https://coveralls.io/github/mattkinsey/bucky?branch=master)

**[Documentation](https://bucky.readthedocs.io/en/latest/)**

**[Developer Guide](https://github.com/mattkinsey/bucky/blob/poetry/dev_readme.md)**

The Bucky model is a spatial SEIR model for simulating COVID-19 at the county level. 

## Getting Started

### Requirements
The Bucky model currently supports Linux and OSX and includes GPU support for accelerated modeling and processing.

* ``git`` must be installed and in your PATH.
* GPU support requires a cupy-compatible CUDA installation. See the [CuPy docs](https://docs.cupy.dev/en/stable/install.html#requirements) for details.

### Installation

Standard installation:
```bash
pip install bucky-covid
```

### Choose a working directory
Bucky will produce multiple folders for historical data and outputs. It's recommended to put these in their own directory, for example ``~/bucky``
```bash
BUCKY_DIR=~/bucky
mkdir $BUCKY_DIR
cd $BUCKY_DIR
```

### Configuration
The default configuration for bucky is located [here](https://github.com/mattkinsey/bucky/tree/master/bucky/base_config). Currently, you can locally modify these options by creating a ``bucky.yml`` in ``BUCKY_DIR`` that will override any of the default options specified in it.

TODO this is WIP and does not work yet:

~~To use a customized configuration you first need to make a local copy of the bucky configuration. In your working directory:~~
```bash
bucky cfg install-local
```

### Download Input Data
To download the required input data to the ``data_dir`` specified in the configuration files (default is ```$(pwd)/data```:
```bash
bucky data sync
```

### Running the Model
To run the model with default settings and produce standard outputs.
```bash
bucky run
```

Equivalently, one can the following command (to provide cli configuration to each part of the process)
```bash
bucky run model
bucky run postprocess
bucky viz plot
```

### CLI options
Each ```bucky``` command has options that can be detailed with the ``--help`` flag. e.g.

    $ bucky run model --help
    
    Usage: bucky run model [OPTIONS]
    
      `bucky run model`, run the model itself, dumping raw monte
      carlo output to raw_output_dir.
    
    Options:
      -d INTEGER         Number of days to project forward
                         [default: 30]
      -s INTEGER         Global PRNG seed  [default: 42]
      -n INTEGER         Number of Monte Carlo iterations  [default:
                         100]
      --runid TEXT       UUID name of current run  [default:
                         2022-06-04__08_00_03]
      --start-date TEXT  Start date for the simulation. (YYYY-MM-DD)
      --help             Show this message and exit.

Further CLI documentation is available in the [documentation](https://docs.buckymodel.com/en/latest/cli.html).
