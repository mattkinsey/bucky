# Usage

## Creating Input Graphs
This section outlines the required structure of the input graph. The Bucky model does not do any data manipulation, smoothing, or correcting to the data it receives from the graph (by design). If data needs to be manipulated or corrected, it should be done before it is placed on the graph.

An example script for creating the input graph for the US is provided in `make_input_graph.py`
Moreover, the OCHA repository https://github.com/OCHA-DAP/pa-COVID-model-parameterization generates input graphs for Afghanistan, the Democratic Republic of Congo, Iraq, Somalia, Sudan and South Sudan.

The graph is created using admin2-level data. If data can not be found at the admin2-level, admin2-level information can be extrapolated using admin2 population and national or state level data (this is expanded upon in the *Population Data* section).

The following data sources are used to create the graph:
* admin2-level shapefile
* admin2-level population data stratified by age
* Historical admin2-level case and death data
* Contact matrix information for the country
* Mobility data (or a proxy)

All data is placed into a single dataframe, joined by the admin2-level key (e.g., FIPS for United States), with the exception of mobility data (which is used to create edges, not nodes).

*NOTE*: This graph should be able to be constructed on the province or state level, but may require model changes. If this is the case, talk to Matt Kinsey.

### Graph-Level Attributes
Administrative information is placed on the top graph-level. For example:
```
'adm1_key' : 'adm1',
'adm2_key' : 'adm2',
'adm1_to_str' : {1 : 'Alabama'}, ...,
'adm0_name': 'US',
'start_date' : '2020-09-25'
```
NOTE: `adm1_to_str` is a dict with key-value pairs indicating the adm1 names for each adm1 value appearing in the graph. 

Contact matrices are also on this level under the key `'contact_mats'`.

### Sample Node
The following is an example node on the graph for the United States. The rest of the documentation will describe what data is necessary to construct this node.

```python
(0,
 {'adm1': 1,
  'Confirmed': 1757.0,
  'Deaths': 25.0,
  'adm2_name': 'Autauga County',
  'N_age_init': array([3364., 3423., 3882., 3755., 3173., 3705., 3461., 3628., 3616.,
         3966., 3811., 3927., 3237., 2589., 2311., 3753.]),
  'Population': 55601.0,
  'IFR': array([6.75414158e-06, 1.24643105e-05, 2.26550214e-05, 4.05345945e-05,
         7.68277004e-05, 1.38382882e-04, 2.54273120e-04, 4.63844627e-04,
         8.51898589e-04, 1.55448599e-03, 2.87077658e-03, 5.20528393e-03,
         9.47735996e-03, 1.73603179e-02, 3.14646839e-02, 9.38331984e-02]),
  'case_hist': array([1207.64227528, 1234.9656055 , 1243.85366911, 1244.13444753,
         1255.27521116, 1268.95333353, 1270.38458817, 1288.05778954,
         1295.55174933, 1297.29129258, 1312.35308192, 1321.2898892 ,
         1323.10534634, 1354.97350342, 1358.88036484, 1362.43488575,
         1377.67551466, 1392.4338964 , 1406.70605635, 1446.3924143 ,
         1450.47616771, 1462.67851762, 1458.98710032, 1470.52271903,
         1481.12998684, 1501.75698721, 1508.06090303, 1513.9178672 ,
         1518.26245703, 1532.99858052, 1553.97101414, 1564.24619451,
         1579.10859377, 1590.56170754, 1597.77332362, 1616.97996262,
         1619.        , 1624.        , 1664.        , 1673.        ,
         1690.        , 1691.        , 1714.        , 1715.        ,
         1738.        , 1757.        ]),
  'death_hist': array([22.76748794, 22.80142062, 22.81307638, 22.79580414, 22.79344408,
         22.79578013, 22.81581338, 22.80532061, 22.7902682 , 22.79603286,
         22.79689139, 22.79601336, 22.79344923, 22.85912123, 22.90405033,
         22.91397178, 22.97898824, 23.02565004, 23.05597481, 23.09719551,
         23.13913548, 24.12323294, 24.17184064, 24.2852927 , 24.38579416,
         24.41284998, 24.41330133, 24.41175889, 24.40910247, 24.41419481,
         24.43286524, 24.47610337, 24.52580854, 24.5245916 , 24.53522989,
         24.54591406, 24.        , 24.        , 24.        , 24.        ,
         24.        , 24.        , 25.        , 25.        , 25.        ,
         25.        ]),
  'adm2': 1001.0})
```

### Population Data
Population data should be at a admin2 level and stratified in 16 5-year age bins (if using Prem et al contact matrices): 

* 0-4 years
* 5-9 years
* 10-14 years
* 15-19 years
* 20-24 years
* 25-29 years
* 30-34 years
* 35-39 years
* 40-44 years
* 45-49 years
* 50-54 years
* 55-59 years
* 60-64 years
* 65-69 years
* 70-74 years
* 75+ years

If population data for an admin2 area is known (i.e. number of total people per admin2), but it is not age-stratified, this data can be extrapolated assuming age-stratified population data exists at some level. For example, assume a country has age-stratified data provided at the national-level. To get the admin2-level age data, the data is separated into the 16 bins (as a 1-dimensional array of length 16). These bins are then normalized by dividing by the sum. Then, the fraction of people living in the admin2 is calculated by dividing admin2 population by the total national population. For each district, this fraction is multiplied by the age vector to produce a admin2-level age vector. This vector is placed on the node under the key *N_age_init*.

The total population for an admin2 is placed on the node under the key *Population*.

### Case Data
Case data should be at the admin2-level and include cumulative data as of the start date of the simulation and historical data for the 45-day period preceding the start date:

* case_hist:  **Cumulative** historical case data
* death_hist :  **Cumulative** historical death data

Historical data is structured as numerical vectors on the node with the keys *case_hist*, *death_hist*. Historical data for every node must have data points for the 45 days preceding the simulation. If there are known errors in the historical data, they must be corrected before being placed on the graph.

### Contact Matrices
Currently, contact matrix data is downloaded from [here](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005697), which has contact matrices for 152 countries. If a country does not appear in this dataset, a country culturally close can be substituted (for example, Pakistan's contact rates were used for Afghanistan), or another dataset can be used. If another dataset is used, the contact matrix must be formatted such that it has the same shape as the number of age demographic bins (i.e. if there are 16 bins, the matrix must be of size 16 x 16).

### Mobility Data
Mobility data is used to construct the edges of the graph. Mobility data, or a proxy for it, is used to describe the contact rates between counties.

The baseline mobility data shows up as an edge attributed called *weight*. *R0_frac* is a factor that is multiplied with the baseline mobility value to model the effect of NPIs, etc., on mobility. For example, given baseline mobility data from February 2020, *R0_frac* would be computed by dividing recent mobility data values with the February 2020 baseline. *R0_frac* exists to provide a knob to tune during the simulation to model NPIs.

## Creating NPI files
While optional, the Bucky model can include Non-Pharmeutical Interventions (NPI) in its simulations. If the user desires these to be included, a .csv-file should be provided that includes the effects of the NPIs.
It should include the columns `admin2,date,elderly_shielding,home,mobility_reduction,other_locations,r0_reduction,school, and work`. 
Each row contains an unique `admin2`-`date` combination. For each other column the value indicates the effect of current NPI's, where values equal to 1 have no effect on the transmission for the given column name. Values smaller than 1 indicate that the NPI's lower the transmission in this domain, while values larger than 1 indicate an increased transmission. 
The columns `home, work, school, other_locations, elderly_shielding` correspond with the columns in the contact matrix. The contact matrix is multiplied with the values of these columns in the NPI file. 
The value in `mobility_reduction` influences the mobility between regions. And the `r0_reduction` applies to the overall community transmission. 

The OCHA repository https://github.com/OCHA-DAP/pa-COVID-model-parameterization implements a method to convert written NPI measures to these numbers. 

## Model
`model.py` takes the following arguments (all are optional as they are either flags or have defined defaults):

* `--graph`, `-g`: Graph file to use. *Default*: Most recently created graph.
* par_file (Positional): Parameter file to use. *Default*: `par/scenario_5.yml`
* `--n_mc`, `-n`: Number of Monte Carlo runs to perform. *Default*: 1000
* `--days`, `-d`: Length of simulation in days. *Default*: 20
* `-o`, `--output` : Output directory for simulation data

Flags:
* `-v`, `--verbose`: Sets the verbosity of the console output
* `-q`, `--quiet`: Suppresses all console output
* `-c`, `--cache`: Cache python files, parameter file, and graph pickle file
* `-nmc`, `--no_mc`: Only performs one run.
* `-gpu`, `--gpu` : Use GPU. *Default* : Uses GPU if cupy is installed

Each Monte Carlo run produces its own `.feather` file. These files are placed in a subdirectory corresponding to the Monte Carlo ID (created by timestamp). 

## Data Postprocessing
*NOTE*: Currently, all postprocessing scripts use the graph file that was used during the simulation. This is used to create a lookup table to specify the relation between admin levels (e.g. admin 2 codes belonging to admin 1). 

### Monte Carlo Aggregation
Before any visualization or analysis can be performed, the raw output needs to be postprocessed. This is done by `postprocess.py`. This script aggregates data at the requested levels (e.g. admin1-level) and produces two types of files: one containing quantiles and one containing mean and standard deviation.

Aggregated files are placed in subfolder named using the Monte Carlo ID within the specified output directory. Filenames are constructed by appending the aggregation level with the aggregation type (quantiles vs mean). For example, the following file contains mean and standard deviation at the national level:
`/output/2020-06-10__14_13_04/adm0_mean_std.csv` 

Arguments:
* file (Positional): Directory location of model-created `.feather` files. *Default*: Most recently created subdirectory in `raw_output/`
* `--graph_file`, `-g`: Graph file used during model. *Default*: Most recently created graph in default graph directoy, `data/input_graphs/`
* `--levels`, `-l`: Aggregation levels. *Default*: adm0, adm1, adm2
* `--quantiles`, `-q`: Quantiles to calculate. *Default*: 0.05, 0.25, 0.5, 0.75, 0.95
* `--output`, `-o`: Directory to place the subfolder containing aggregated files. *Default*: `output/`
* `--prefix`: Prefix for the subfolder name. *Default*: None
* `--end_date`: If a user passes in an end date, aggregated data will not include data past this point. *Default*: None
* `--verbose`, `-v`: Prints extra information during postprocessing.
* `--no_quantiles`: Skips computing quantiles and computes only mean and standard deviation.
* `--lookup`: Pass in an explicit lookup table for geographic mapping info. See below caveat.

#### Lookup Tables
By default, postprocessing uses geographic information on the graph to aggregate geographic areas. For special cases, a lookup table may be passed in via the `--lookup` command. This is intended to be used for splitting states into non-FIPS divisions. When a lookup table is passed in, the output directory will be prepended with a string to distinguish it from output created from the same simulation using the graph file.


### Visualization
The Bucky model has the capability to create two types of visualization: plots and maps. Both are contained within the `viz/` directory. 

All visualizations are placed in subfolders in the same directory as the aggregated directory. Plots are placed in `plots/` and maps are placed in `maps/`. These folders can be renamed with command-line arguments, but will still be placed within the aggregated data folder.

Example:
```
2020-07-28__15_21_52/
├── adm0_mean_std.csv
├── adm0_quantiles.csv
├── adm1_mean_std.csv
├── adm1_quantiles.csv
├── adm2_mean_std.csv
├── adm2_quantiles.csv
├── maps
│   └── ADM1
│       ├── adm1_AlabamaDailyReportedCases2020-07-26.png
│       ├── adm1_AlabamaDailyReportedCases2020-08-02.png
│       ├── ...
└── plots
    ├── ADM1
    │   ├── Alabama.png
    │   ├── ...
    ├── US.csv
    └── US.png
```

#### Plots

Plots can be created at any of the three admin levels. Each plot contains two subplots. These plots can optionally include historical data. Example usage: 
```console
python3 viz/plot.py -i output/2020-06-10__14_13_04/ --hist 
```
*NOTE*: By default, `plot.py` makes admin0 and admin1 plots. In order to create admin2-level plots, a user must also pass in an admin1 name. For example, to create county-level plots for Arizona, `--adm1_name Arizona` must be passed in as an argument.

Arguments and flags:
* `--input_dir`, `-i`: Directory location of aggregated data. *Default*: Most recently created subdirectory in the default directory for processed data (`output/`)
* `--output`, `-o`: Output directory for plots. *Default*: `$INPUT_DIR/plots/`
* `--graph_file`, `-g`: Graph file used during model. *Default*: Most recently created graph in default graph directoy, `data/input_graphs/`
* `--levels`, `-l`: Requested plot levels. *Default*: adm0, adm1
* `--hist`, `-hist`: Plot historical data in addition to simulation data.
* `--window_size`, `-w`: Size of window (in days) to apply to historical data. *Default*: 7
* `--plot_columns`: Columns to plot. *Default*: daily_cases_reported, daily_deaths
* `--hist_columns`: Historical columns to plot. Note: If only one historical column is passed in, historical data will only be present on the top plot. *Default*: Confirmed, Deaths
* `--hist_start`: Start date of historical data. If not passed in, will align with start date of simulation
* `--hist_file`: History file to use. *Default*: Will use US CSSE historical data
* `--end_date`: If a user passes in an end date, data will not be plotted past this point. 
* `--adm1_name`: Name of adm1 to create adm2 plots for. *Default*: None
* `--lookup`: Pass in an explicit lookup table for geographic mapping info
* `-v`, `--verbose` : Prints extra information during plotting

By default, confidence intervals are plotted using quantiles. Optionally, the standard deviation can be used by passing in the following arguments:
* `--n_mc`, `-n`: Number of Monte Carlo runs from simulation. *Default*: 1000
* `--use_std`: Flag to indicate standard deviation should be used instead of quantiles.

##### Historical Data
The plotting utility expects historical data to be at the adm2-level.

#### Maps
Maps are created at the adm0 or adm1 level. In order to create maps, shapefiles must be provided one level down from the requested map (e.g. adm2-level shapefile must be provided for adm1-level maps). Maps can be created for specific dates or distributed throughout the length of the simulation with a requested frequency.

Example usage: 
```console
python3 viz/map.py -i output/2020-06-10__14_13_04/  --all_adm1--adm2_shape data/shapefiles/tl_2019_us_county.shp --dates 2020-06-01
```

Arguments and flags:
* `--input_dir`, `-i`: Directory location of processed simulation data, *Default*: Most recently created subdirectory in default output directory (`output/`)
* `--output`, `-o`: Output directory for maps. *Default*: `$INPUT_DIR/maps/`
* `--graph_file`, `-g`: Graph file used during model. *Default*: Most recently created graph in default graph directoy, `data/input_graphs/`
* `--adm0`: Create adm0-level plot 
* `--adm1`: Create adm1-level plot for the requested adm1 name
* `--all_adm1`: Create adm1-level plot for every available adm1-level area.
* `--adm1_shape`: Location of adm1 shapefile. *Default*: `data/shapefiles/tl_2019_us_state.shp`
* `--adm2_shape`: Location of adm2 shapefile. *Default* : `data/shapefiles/tl_2019_us_county.shp`
* `--columns`: Data columns to plot. Maps are created separately for each requested column. *Default*: daily_cases_reported, daily_deaths 
* `--dates`, `-d`: Dates to create maps. If passed in, takes priority over `frequency`. *Default*: None
* `--freq`, `-f`: Frequency at which to create maps. Allowed values: daily, weekly, monthly. *Default*: weekly
* `--mean`: Use mean value instead of median value for map
* `--linear`: Use linear scaling for values instead of log
* `--cmap`, `-c`: Colormap to use. Must be a valid matplotlib colormap. *Default*: Reds
* `--lookup`: Pass in an explicit lookup table for geographic mapping info
