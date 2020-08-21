# Installation Guide
## Environment Setup
* Tested with Ubuntu 18.04.4

1. Install anaconda by executing the *.sh* file, see Matt K. for the correct anaconda version.
2. SSL verification needs to be turned off for anaconda, otherwise packages will not be downloaded/installed properly.
Therefore, run the following command, which creates a .condarc file in $HOME: 
```console
conda config --set ssl_verify no
```
3. Within $HOME, create a .pip/pip.conf file and add the following code, which effectively turns off SSL for pip:
```python
[global]
trusted-host = pypi.python.org
               pypi.org
               files.pythonhosted.org
```
4. Install gcc and g++ using sudo apt install.
5. Create the covid environment, which assumes the user has the covid.yml file (This command builds an environment directory and stores it in anaconda3/envs/):
```console
conda env create -f environment.yml
```
6. Activate the environment:
```console
conda activate *environment_name*
```
7. Deactivate the environment:
```console
conda deactivate
```

## Data Setup
The included script `get_us_data.sh` downloads required datasets, including git repositories for mobility and case data.

### US Data Sources

* COVID-19 Data Repository by the Center for Systems Science and Engineering at Johns Hopkins University
    * COVID-19 Case and death data on the county level
    * [Git repository](https://github.com/CSSEGISandData/COVID-19)
* Descartes Labs: Data for Mobility Changes in Response to COVID-19
    * State and county-level mobility statistics
    * [Git repository](https://github.com/descarteslabs/DL-COVID-19)
* COVID Exposure Indices from PlaceIQ movement data
    * State and county-level location exposure indices
    * Reference: *Measuring movement and social contact with smartphone data: a real-time application to COVID-19* by Couture, Dingel, Green, Handbury, and Williams [Link](https://github.com/COVIDExposureIndices/COVIDExposureIndices/blob/master/CDGHW.pdf)
    * [Git repository](https://github.com/COVIDExposureIndices/COVIDExposureIndices)
* The COVID Tracking Project at The Atlantic
    * COVID-19 case and death data at the state level
    * [Git repository](https://github.com/COVID19Tracking/covid-tracking-data)
* US TIGER shapefiles from the US Census 
    * [Description](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
* US Census Bridged-Race Population estimates
    * [Description](https://www.cdc.gov/nchs/nvss/bridged_race/Documentation-Bridged-PostcenV2018.pdf)
* Social Contact Matrices for 152 Countries
    * *Projecting social contact matrices in 152 countries using contact surveys and demographic data*, Prem et al. [Link to paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005697)
* USAFacts Coronavirus Stats and Data
    * County-level coronavirus cases and deaths
    * [Link](https://usafacts.org/issues/coronavirus/)

