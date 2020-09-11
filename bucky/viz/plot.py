import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm

from ..util.read_config import bucky_cfg
from ..util.readable_col_names import readable_col_names
from .geoid import read_geoid_from_graph, read_lookup

plt.style.use("ggplot")

# Historical data locations
DEFAULT_HIST_FILE = os.path.join(
    bucky_cfg["data_dir"], "cases/csse_hist_timeseries.csv"
)
WINDOWED_FILE = os.path.join(
    bucky_cfg["data_dir"], "cases/csse_windowed_timeseries.csv"
)

parser = argparse.ArgumentParser(description="Bucky model plotting tools")

# Location of processed data
parser.add_argument(
    "-i",
    "--input_dir",
    default=max(
        glob.glob(os.path.join(bucky_cfg["output_dir"], "*/")),
        key=os.path.getctime,
        default="Most recently created folder in output_dir",
    ),
    type=str,
    help="Directory location of aggregated data",
)

# Output directory
parser.add_argument(
    "-o",
    "--output",
    default=None,
    type=str,
    help="Output directory for plots. Defaults to input_dir/plots/",
)

# Graph file used for this run. Defaults to most recently created
parser.add_argument(
    "-g",
    "--graph_file",
    default=None,
    type=str,
    help="Graph file used during model. Defaults to most recently created graph",
)

# Aggregation levels, e.g. state, county, etc.
parser.add_argument(
    "-l",
    "--levels",
    default=["adm0", "adm1"],
    nargs="+",
    type=str,
    help="Requested plot levels",
)

# Columns for plot, historical data
default_plot_cols = ["daily_cases_reported", "daily_deaths"]
default_hist_cols = ["Confirmed_daily", "Deaths_daily"]
parser.add_argument(
    "--plot_columns",
    default=default_plot_cols,
    nargs="+",
    type=str,
    help="Columns to plot",
)
parser.add_argument(
    "--hist_columns",
    default=default_hist_cols,
    nargs="+",
    type=str,
    help="Historical columns to plot",
)

# Specify number of MC runs
parser.add_argument(
    "-n",
    "--n_mc",
    default=1000,
    type=int,
    help="Number of Monte Carlo runs that were performed during simulation",
)

# Can pass in a lookup table to use in place of graph
parser.add_argument(
    "--lookup", default=None, type=str, help="Lookup table for geographic mapping info",
)

# Pass in a specific historical start date and historical file
parser.add_argument(
    "--hist_start",
    default=None,
    type=str,
    help="Start date of historical data. If not passed in, will align with start date of simulation",
)
parser.add_argument(
    "--hist_file",
    default=None,
    type=str,
    help="File to use for historical data if not using US CSSE data",
)

# Optional flags
parser.add_argument(
    "--adm1_name", default=None, type=str, help="Admin1 to make admin2-level plots for",
)
parser.add_argument(
    "--end_date",
    default=None,
    type=str,
    help="Data will not be plotted past this point",
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Print extra information"
)
parser.add_argument(
    "--use_std",
    action="store_true",
    help="Flag to indicate standard deviation should be used instead of quantiles",
)
parser.add_argument(
    "-hist",
    "--hist",
    action="store_true",
    help="Plot historical data in addition to simulation data",
)

parser.add_argument(
    "-q",
    "--extra_quantiles",
    action="store_true",
    help="Indicate that more than 5 quantiles should be plotted")

# Size of window in days
parser.add_argument(
    "-w",
    "--window_size",
    default=7,
    type=int,
    help="Size of window (in days) to apply to historical data",
)


def add_daily_history(history_data, window_size=None):
    """Applies a window to cumulative historical data to get daily data.
    
    Parameters
    ----------
    history_data : Pandas DataFrame
        Cumulative case and death data
    window_size : int or None
        Size of window in days

    Returns
    -------
    history_data : Pandas DataFrame
        Historical data with added columns for daily case and death data
    """
    # TODO: Correct territory data by distributing
    # history_data['adm2'].replace(66, 66010, inplace=True)
    # history_data['adm2'].replace(69, 69110, inplace=True)

    history_data = history_data.set_index(["adm2", "date"])
    history_data.sort_index(inplace=True)

    # Remove string columns if they exist
    str_cols = list(history_data.select_dtypes(include=["object"]).columns)
    history_data = history_data.drop(columns=str_cols)

    daily_data = history_data.groupby(level=0).diff()
    daily_data.columns = [str(col) + "_daily" for col in daily_data.columns]

    if window_size is not None:

        daily_data = (
            daily_data.reset_index(level=0)
            .groupby("adm2")
            .rolling(window_size, min_periods=window_size // 2, center=True)
            .mean()
            .drop(columns=["adm2"])
        )

        # daily_data.reset_index().set_index('adm2', inplace=True)
        # daily_data = daily_data.rolling(7, center=True, on='date').mean()
        # daily_data = hdaily_data.set_index(['adm2', 'date'])

    history_data = history_data.merge(daily_data, left_index=True, right_index=True)
    history_data.reset_index(inplace=True)

    return history_data


def interval(mean, sem, conf, N):
    z = scipy.stats.t.ppf((1 + conf) / 2.0, N - 1)
    return (mean - sem * z).clip(lower=0.0), mean + sem * z


def plot(
    output_dir,
    lookup_df,
    key,
    sim_data,
    hist_data,
    plot_columns,
    hist_columns,
    use_std=False,
    n_mc=None,
    extra_quantiles=False
):
    """Given a dataframe and a key, creates plots with requested columns.

    For example, a DataFrame with state-level data would create a plot for
    each unique state. Simulation data is plotted as a line with shaded
    confidence intervals. Historical data is added as scatter points if
    requested. 

    Parameters
    ----------
    output_dir : string
        Location to place created plots.
    lookup_df : Pandas DataFrame
        Dataframe containing information relating different geographic
        areas
    key : string
        Key to use to relate simulation data and geographic areas. Must
        appear in lookup and simulation data (and historical data if 
        applicable)
    sim_data : Pandas DataFrame
        Simulation data to plot
    hist_data : Pandas DataFrame
        Historical data to add to plot
    plot_columns : list of strings
        Simulation columns to plot
    hist_columns : list of strings
        Historical data columns to plot
    use_std : boolean, default=False
        Indicating whether standard deviation should be used instead of 
        confidence intervals
    n_mc : int
        Number of Monte Carlos performed for this simulation. Required if
        using standard deviation instead of confidence intervals.
    extra_quantiles : bool
        If true, more than 5 quantiles will be plotted
    """

    # Need N to plot standard dev
    if use_std and not n_mc:
        print("Error: Need number of Monte Carlo runs to plot standard deviation")
        return

    # Drop lookup nans
    lookup_df = lookup_df.dropna()

    # Make one plot for each area
    # For state plots, one plot per state
    # For counties, one plot per county
    unique_sim_areas = sim_data[key].unique()
    unique_lookup_areas = lookup_df[key].unique()

    # Some lookups have a subset. Use whichever set of areas is smaller
    if len(unique_sim_areas) < len(unique_lookup_areas):
        unique_areas = unique_sim_areas
    else:
        unique_areas = unique_lookup_areas

    for area in tqdm(unique_areas, total=len(unique_areas)):

        # Get name
        name = lookup_df.loc[lookup_df[key] == area][key + "_name"].values[0]

        # If admin2-level, get admin1-level as well
        if key == "adm2":

            admin1_name = lookup_df.loc[lookup_df[key] == area]["adm1_name"].values[0]
            name = admin1_name + " : " + name

        # Get data for this area
        area_data = sim_data.loc[sim_data[key] == area]
        area_data = area_data.assign(date=pd.to_datetime(area_data["date"]))

        # Initialize figure
        fig, axs = plt.subplots(
            2,
            sharex=True,
            gridspec_kw={"hspace": 0.1, "height_ratios": [2, 1]},
            figsize=(15, 7),
        )
        fig.suptitle(name.upper())

        for i, col in enumerate(plot_columns):

            # Data is either quantiles or mean/std
            if not use_std:

                # Get number of quantiles
                quantiles = area_data["q"].unique()
                num_intervals = len(quantiles)

                # make sure they're sorted
                quantiles.sort()

                # Set index
                area_data.set_index(["date", "q"], inplace=True)

                # Middle is median
                median_data = area_data.xs(quantiles[int(num_intervals / 2)], level=1)[
                    col
                ]
                dates = median_data.index.values

                if extra_quantiles:

                    # Plot median and outer quantiles
                    median_data.plot(
                        linewidth=1.5, color='k', alpha=0.75, label=readable_col_names[col], ax=axs[i]
                    )

                    # Iterate over pairs of quantiles
                    num_quantiles = len(quantiles)

                    # Scale opacity
                    alpha = 1. / (num_quantiles  // 2)
                    for q in range(num_quantiles // 2):

                        lower_q = quantiles[q]
                        upper_q = quantiles[num_quantiles - 1 - q]

                        lower_data = area_data.xs(lower_q, level=1)[col]
                        upper_data = area_data.xs(upper_q, level=1)[col]

                        axs[i].fill_between(
                            dates,
                            lower_data,
                            upper_data,
                            linewidth=0,
                            alpha=alpha,
                            color="b",
                            interpolate=True
                        )

                else:
                    # Plot median and outer quantiles
                    median_data.plot(
                        linewidth=2.75, label=readable_col_names[col], ax=axs[i]
                    )

                    # Grab other quantiles in pairs
                    outer_quantiles = [quantiles[0], quantiles[-1]]
                    oq_lower = area_data.xs(outer_quantiles[0], level=1)[col]
                    oq_upper = area_data.xs(outer_quantiles[1], level=1)[col]
                    outer_label = (
                        str(outer_quantiles[0] * 100)
                        + "% - "
                        + str(outer_quantiles[1] * 100)
                        + "% CI"
                    )

                    axs[i].fill_between(
                        dates,
                        oq_lower,
                        oq_upper,
                        linewidth=0,
                        alpha=0.2,
                        color="0.2",
                        interpolate=True,
                        label=outer_label,
                    )

                    # If there are more than 3, plot the inner quantiles as well
                    if num_intervals > 3:
                        inner_quantiles = [quantiles[1], quantiles[-2]]
                        iq_lower = area_data.xs(inner_quantiles[0], level=1)[col]
                        iq_upper = area_data.xs(inner_quantiles[1], level=1)[col]

                        inner_label = (
                            str(inner_quantiles[0] * 100)
                            + "% - "
                            + str(inner_quantiles[1] * 100)
                            + "% CI"
                        )
                        axs[i].fill_between(
                            dates,
                            iq_lower,
                            iq_upper,
                            linewidth=0,
                            alpha=0.2,
                            color="r",
                            interpolate=True,
                            label=inner_label,
                        )

                # axs[i].set_ylim([.8*oq_lower.min(), 1.2*oq_upper.max()])

                area_data.reset_index(inplace=True)

            # Plot mean/standard deviation
            else:

                # Set index
                area_data.set_index("date", inplace=True)

                # Get mean, std, and dates
                mu = area_data.loc[area_data["stat"] == "mean"][col]
                dates = mu.index.values
                sem = area_data.loc[area_data["stat"] == "std"][col]

                # Compute intervals
                col_25, col_75 = interval(mu, sem, 0.5, n_mc)

                col_05, col_95 = interval(mu, sem, 0.95, n_mc)

                # Plot data
                mu.plot(linewidth=2.75, label=readable_col_names[col], ax=axs[i])
                axs[i].fill_between(
                    dates,
                    col_05,
                    col_95,
                    linewidth=0,
                    alpha=0.2,
                    color="0.2",
                    interpolate=True,
                    label="2.5% - 97.5% CI",
                )
                axs[i].fill_between(
                    dates,
                    col_25,
                    col_75,
                    linewidth=0,
                    alpha=0.2,
                    color="r",
                    interpolate=True,
                    label="25% - 75% CI",
                )

                area_data.reset_index(inplace=True)

            # Plot historical data which is already at the correct level
            if hist_data is not None and i <= (len(hist_columns) - 1):

                actuals = hist_data.loc[hist_data[key] == area]

                if not actuals.empty:

                    actuals = actuals.assign(date=pd.to_datetime(actuals["date"]))
                    # actuals.set_index('date', inplace=True)
                    actuals.plot.scatter(x="date", y=hist_columns[i], ax=axs[i], color='r')

                    # Set xlim
                    axs[i].set_xlim(actuals["date"].min(), dates.max())
                    # axs[i].scatter(actual_dates[ind:], actual_vals[ind:], label='Historical data', color='b')
                else:
                    print("Historical data missing for: " + name)

            axs[i].grid(True)
            axs[i].legend()
            axs[i].set_ylabel("Count")

        plot_filename = os.path.join(output_dir, name + ".png")
        plot_filename = plot_filename.replace(" : ", "_")
        plot_filename = plot_filename.replace(" ", "")
        area_data.to_csv(plot_filename.replace(".png", ".csv"))
        plt.savefig(plot_filename)
        plt.close()


def make_plots(
    adm_levels,
    input_dir,
    output_dir,
    lookup,
    plot_hist,
    plot_columns,
    hist_columns,
    use_std,
    N,
    use_windowed,
    end_date,
    admin1=None,
    hist_start=None,
    hist_file=None,
    extra_quantiles=False
):
    """Wrapper function around plot. Creates plots, aggregating data if necessary.

    Parameters
    ----------
    adm_levels : list of strings
        List of ADM levels to make plots for
    input_dir : string
        Location of simulation data
    output_dir : string
        Parent directory to place created plots. 
    lookup : Pandas DataFrame
        Lookup table for geographic mapping information
    plot_hist : boolean
        If true, will plot historical data
    plot_columns : list of strings
        List of columns to plot from simulation data
    hist_columns : list of strings
        List of columns to plot from historical data
    use_std : boolean
        If true, use standard deviation instead of quantiles for
        confidence intervals
    N : int
        Number of Monte Carlo runs from simulation
    use_windowed : boolean
        Use windowed or cumulative historical data
    end_date : string, formatted as YYYY-MM-DD
        Plot data until this date
    admin1 : list of strings, or None
        List of admin1 values to make plots for. If None, a plot will be
        created for every unique admin1 values. Otherwise, plots are only
        made for those requested.
    hist_start : string, formatted as YYYY-MM-DD
        Plot historical data from this point. If None, aligns with
        simulation start date
    hist_file : string or None
        File to use for historical data. If None, uses default defined at
        top of file.
    extra_quantiles : bool
        If true, more than 5 quantiles will be plotted
    """
    # Loop over requested levels
    for level in adm_levels:

        if use_std:
            filename = level + "_mean_std.csv"
        else:
            filename = level + "_quantiles.csv"

        # Read the requested file from the input dir
        data = pd.read_csv(os.path.join(input_dir, filename))
        data = data.assign(date=pd.to_datetime(data["date"]))

        # Get dates for data
        start_date = data["date"].min()
        if end_date is not None:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = data["date"].max()

        # Drop data not within range
        data = data.loc[(data["date"] < end_date) & (data["date"] > start_date)]

        # Determine if sub-folder is necessary (state or county-level)
        plot_dir = output_dir
        if level == "adm2" or level == "adm1":

            plot_dir = os.path.join(plot_dir, level.upper())

            # create directory
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

        # For admin2 only: if a admin1 name was passed in, only keep data within that admin1
        if level == "adm2" and admin1 is not None:

            admin2_vals = lookup_df.loc[lookup_df["adm1_name"] == admin1][
                "adm2"
            ].unique()
            data = data.loc[data["adm2"].isin(admin2_vals)]

        # Read historical data if needed
        level_hist_data = None
        if plot_hist:

            # Read historical data
            if hist_file is not None:
                hist_data = pd.read_csv(hist_file)
            else:
                hist_data = pd.read_csv(DEFAULT_HIST_FILE)

            # If admin2 key is FIPS, rename
            if "FIPS" in hist_data.columns:
                hist_data.rename(columns={"FIPS": "adm2"}, inplace=True)

            # Add daily data
            hist_data = add_daily_history(hist_data, use_windowed)

            # Aggregate if necessary
            if level == "adm1":

                # Historical data is at the admin2-level, so aggregate
                lookup_df.set_index("adm2", inplace=True)
                level_dict = lookup_df["adm1"].to_dict()

                # Map historical data to values in lookup
                # in multiple steps to drop counties that don't appear in lookup
                hist_data["adm1"] = hist_data["adm2"].map(level_dict)
                hist_data = hist_data.dropna(subset=["adm1"])

                # Groupby admin1 and date and sum
                level_hist_data = (
                    hist_data.groupby(["date", "adm1"]).sum().reset_index()
                )

            if level == "adm0":
                # No need for mapping, all data is in this country
                # Groupby date and date and sum
                level_hist_data = hist_data.groupby("date").sum().reset_index()
                level_hist_data = level_hist_data.assign(adm0="US")

            if level == "adm2":

                # No aggregation required
                level_hist_data = hist_data

            # Drop data not within requested time range
            level_hist_data = level_hist_data.assign(
                date=pd.to_datetime(level_hist_data["date"])
            )

            if hist_start is not None:
                level_hist_data = level_hist_data.loc[
                    (level_hist_data["date"] < end_date)
                    & (level_hist_data["date"] > hist_start)
                ]
            else:
                level_hist_data = level_hist_data.loc[
                    (level_hist_data["date"] < end_date)
                    & (level_hist_data["date"] > start_date)
                ]

        plot(
            output_dir=plot_dir,
            lookup_df=lookup_df,
            key=level,
            sim_data=data,
            hist_data=level_hist_data,
            plot_columns=plot_columns,
            hist_columns=hist_columns,
            use_std=use_std,
            n_mc=N,
            extra_quantiles=extra_quantiles
        )


if __name__ == "__main__":

    # Parse CLI args
    args = parser.parse_args()
    print(args)

    # Parse arguments
    input_dir = args.input_dir

    output_dir = args.output

    if output_dir is None:
        output_dir = os.path.join(input_dir, "plots")

    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # aggregation levels
    levels = args.levels

    # Columns for plotting
    plot_cols = args.plot_columns
    hist_cols = args.hist_columns

    if args.lookup is not None:
        lookup_df = read_lookup(args.lookup)
    else:
        lookup_df = read_geoid_from_graph(args.graph_file)

    # Historical data start
    hist_start = args.hist_start
    hist_file = args.hist_file

    # Parse optional flags
    window = args.window_size
    N = args.n_mc
    plot_historical = args.hist
    verbose = args.verbose
    use_std = args.use_std
    end_date = args.end_date
    extra_quantiles = args.extra_quantiles

    # Plot
    make_plots(
        levels,
        input_dir,
        output_dir,
        lookup_df,
        plot_historical,
        plot_cols,
        hist_cols,
        use_std,
        N,
        window,
        end_date,
        args.adm1_name,
        hist_start,
        hist_file,
        extra_quantiles
    )
