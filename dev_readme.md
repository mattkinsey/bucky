# Bucky Developer Guide
WIP document on getting the dev env setup

|This only applies to the ``poetry`` branch until all this is merged into master!|
|---------------------------------------------------------------------------------

## Requirements
### System
You need at least git, python 3.7+ and standard build tools (like ubuntu's build-essential)

TODO someone needs to run through an install in a fresh env to make sure there's nothing else here...

### Install [Poetry](https://python-poetry.org/)
Download [install-poetry.py](https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py) and run it:
``` bash
wget https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py
python install-poetry.py
```

#### Mac Requirements
If installing Bucky on a Mac, the following dependencies must also be installed. *Note*: This assumes you have brew installed.

- OpenBLAS
`brew install openblas`

- GDAL
`brew install gdal`

- PyProj
`brew install proj`

### Create your own fork
To submit pull requests to the [main repository](https://github.com/mattkinsey/bucky) you'll need to setup your own fork of the repo:


* In the GitHub UI, [fork the main repository](https://help.github.com/articles/fork-a-repo/) to your own account.
* Add your fork of the repository as the origin
  ``` bash
  git clone https://github.com/[your_username]/bucky.git -b [branch_name]
  cd bucky
  ```
* Add the original version of the repository as an "upstream" source:
  ```bash
  git remote add upstream https://github.com/mattkinsey/bucky.git
  ```

Do all of your work in this fork, make a bunch of commits, push to it whenever you want, etc. Once a feature is done, use the GitHub UI to create a pull request from your fork to the upstream repository, so that it may be code reviewed.

### Install Nox (optional)
If you want to run the full test suite against multiple versions of python (3.7, 3.8, 3.9, and 3.10), you'll need to setup Nox+pyenv. 

| Pyenv potentially mucks up your python setup SYSTEM WIDE, be careful and only do this if you need to. |
|-------------------------------------------------------------------------------------------------------|

* Install [pyenv](https://nox.thea.codes/).

  See [this guide](https://amaral.northwestern.edu/resources/guides/pyenv-tutorial) for details.

    * If installing Bucky on a Mac, use brew to install.
          * `brew install pyenv`
          * After installing pyenv, add to your path by adding the following to your .zshrc (or bashrc) file:

```
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

* Install compatible python versions in pyenv (this will take awhile to build each one).
  ```bash
  pyenv install 3.7.11
  pyenv install 3.8.12
  pyenv install 3.9.10
  pyenv install 3.10.2
  ```

* Set the local python versions for bucky:
  ```bash
  cd bucky
  pyenv local 3.10.2 3.9.10 3.8.12 3.7.11
  ```
  The first version listed is the one used when you run plain ``python``. Every other version can be used by invoking ``python<major.minor>``. For example, use ``python3.7`` to invoke Python 3.7.

* Install [Nox](https://nox.thea.codes/) at the system level (not in a bucky venv)
  ```bash
  pip install nox
  ```

Now you should be setup to run the full test suite against all four python versions by running nox in the top level bucky directory:
```bash
cd bucky
nox
```

### Install GDAL
Some dependencies (geopandas, fiona) require GDAL to be installed at the system level. How to do this depends on your OS. The following are untested (except arch):

#### Arch
```bash
sudo pacman -Sy gdal
```

#### Ubuntu
```bash
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev
```

#### OS X
```bash
brew install gdal
```

#### Windows
TODO You will need to manually install the gdal/fiona wheels under windows.


### Install Python Dev requirements
In your local checkout of your bucky fork (you should be in the top level bucky folder that contains ``pyproject.toml``), install the local development requirements inside a virtual environment using poetry:
```bash
poetry install
poetry run inv install-hooks
```
NB: you can install the GPU extras version on compatiable systems using ``poetry install -E gpu`` instead.

Everything should be good to go now, you can execute commands in the bucky venv using ``poetry run <cmd>``

**For Mac users:** Because some dependencies were installed via brew, their locations must be specified when running poetry install. Use the following command:

`OPENBLAS="$(brew --prefix openblas)" DYLD_FALLBACK_LIBRARY_PATH="$(HOME)/lib:/usr/local/lib:/lib:/usr/lib" poetry install`

Other poetry commands can be run as normal.

## Usage
TODO need more info here

You should now be able to execute the bucky cli using the editable install in the poetry venv:
```bash
poetry run bucky
```

### Writing tests

### Invoke tasks
There are some predefined devel tasks that can be run using [Invoke](https://www.pyinvoke.org/) (which is already installed as a dev dep).

#### Clean
* ``poetry run inv clean`` - remove ALL temporary files
  * ``poetry run inv clean_build`` - remove temporary files from building wheels
  * ``poetry run inv clean_python`` - remove temporary python caches
  * ``poetry run inv clean_tests`` - remove temporary files from testing
  * ``poetry run inv clean_docs`` - remove temporary files from ``sphinx-build``ing the docs

#### Linting
* ``poetry run inv hooks`` - run all pre-commit hooks, (like running ``pre-commit run -a`` in the poetry venv.

* ``poetry run inv lint`` - run all linters
  * ``poetry run inv format`` - run black and isort
  * ``poetry run inv flake8`` - run flakehell (which wraps flake8 and pylint)
  * ``poetry run inv safety`` - run safety to check for outdated/insecure dependences (dependabot should catch all these though :shrug:)

#### Testing
* ``poetry run inv tests`` - run pytest on the active python version (see [here](#install-nox-optional) for running against all python versions)

* ``poetry run inv coverage`` - generate coverage report

#### Documentation
* ``poetry run inv docs`` - build a local copy of the sphinx docs

#### Misc
* ``poetry run inv version`` - bump the version and commit

## Building the wheels
Don't worry about this unless you're matt...
## Pypi
Don't worry about this unless you're matt...
