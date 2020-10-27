# Usage

## Creating Input Graphs
`make_input_graph.py` creates input graphs used by the model and by certain postprocessing steps. This script has one required command line argument: `--date`. This field specifies the **last** date of historical data to use. This is also the **start** date of the simulation.

Arguments and flags:
* `-d`, `--date` (**Required**): Start date of simulaiton. Format: YYYY-MM-DD
* `-o`, `--output` : Output directory for created graph file. *Default*: `data/input_graphs/`
* `--hist_file` : Specify historical case file. *Default:* Uses CSSE data. **Note**: Must be county-level 
* `--no_update` : Skips updating data repositories 

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
