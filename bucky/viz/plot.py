"""Creates line plots with confidence intervals at the ADM0, ADM1, or ADM2 level."""
import multiprocessing
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import pandas as pd
import tqdm
import us
from loguru import logger

from .get_historical_data import get_historical_data, get_historical_fit
from .readable_col_names import readable_col_names


def get_all_plot_titles(l_table, adm_key, adm_values):
    """Determine plot titles based on a list of adm codes and lookup table.

    For ADM0 and ADM1 plots, uses names. For ADM2 plots, the ADM1 name is
    included in addition to the ADM2 name.

    Parameters
    ----------
    l_table : pandas.DataFrame
        Dataframe containing information relating different geographic
        areas
    adm_key : str
        Admin level key
    adm_value : int
        Admin code value for area

    Returns
    -------
    all_plot_titles : list
        List of formatted string to use for plot titles
    """

    all_plot_titles = []

    for val in adm_values:
        plot_title = l_table.loc[l_table[adm_key] == val][adm_key + "_name"].values[0]

        # If admin2-level, get admin1-level as well
        if adm_key == "adm2":

            admin1_name = l_table.loc[l_table[adm_key] == val]["adm1_name"].values[0]
            plot_title = admin1_name + " : " + plot_title

        all_plot_titles.append(plot_title)

    return all_plot_titles


def get_plot_title(l_table, adm_key, adm_value):
    """Determine plot title for a given area based on adm code and lookup table.

    For ADM0 and ADM1 plots, uses names. For ADM2 plots, the ADM1 name is
    included in addition to the ADM2 name.

    Parameters
    ----------
    l_table : pandas.DataFrame
        Dataframe containing information relating different geographic
        areas
    adm_key : str
        Admin level key
    adm_value : int
        Admin code value for area

    Returns
    -------
    plot_title : str
        Formatted string to use for plot title
    """
    plot_title = l_table.loc[l_table[adm_key] == adm_value][adm_key + "_name"].values[0]

    # If admin2-level, get admin1-level as well
    if adm_key == "adm2":

        admin1_name = l_table.loc[l_table[adm_key] == adm_value]["adm1_name"].values[0]
        plot_title = admin1_name + " : " + plot_title

    return plot_title


def add_col_data_to_plot(axis, col, sim_data, quantiles_list):
    """Adds simulation data for requested column to initialized matplotlib axis object.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Previously initialized axis object
    col : str
        Column to add to plot
    sim_data : pandas.DataFrame
        Area simulation data
    quantiles_list : list of float, or None
        List of quantiles to plot. If None, will plot all available
        quantiles in data.

    Returns
    -------
    axis : matplotlib.axes.Axes
        Modified axis object with added data
    """
    # Set index
    sim_data = sim_data.set_index(["date", "quantile"])

    # Middle is median
    num_q = len(quantiles_list)
    median_data = sim_data.xs(quantiles_list[int(num_q / 2)], level=1)[col]
    dates = median_data.index.values

    # Plot median and outer quantiles
    median_data.plot(
        linewidth=1.5,
        color="k",
        alpha=0.75,
        label=readable_col_names[col],
        ax=axis,
    )

    # Iterate over pairs of quantiles
    num_quantiles = len(quantiles_list)

    # Scale opacity
    alpha = 1.0 / (num_quantiles // 2)
    for q in range(num_quantiles // 2):

        lower_q = quantiles_list[q]
        upper_q = quantiles_list[num_quantiles - 1 - q]

        lower_data = sim_data.xs(lower_q, level=1)[col]
        upper_data = sim_data.xs(upper_q, level=1)[col]

        axis.fill_between(
            dates,
            lower_data,
            upper_data,
            linewidth=0,
            alpha=alpha,
            color="b",
            interpolate=True,
        )

    sim_data = sim_data.reset_index()

    return axis


def get_group_historical_data(historical_data, adm_key, adm_values):
    """Attempts to get historical data for each adm value in a list, in order.

    Parameters
    ----------
    historical_data : pandas.DataFrame
        Dataframe of historical data
    adm_key : str
        Admin key to use to relate simulation data and geographic areas
    adm_values : list
        List of Admin code values for which to fetch data

    Returns
    -------
    grouped_historical_data : list of pandas.DataFrames
        List of historical data in same order as input list of admin values
    """

    grouped_historical_data = []

    for val in adm_values:

        # Find historical data for this location and add to list
        data = historical_data.loc[historical_data[adm_key] == val]
        grouped_historical_data.append(data)

    return grouped_historical_data


def add_hist_data_to_plot(axis, historical_data, adm_key, adm_value, col, sim_max_date):
    """Add historical data for requested column and area to initialized matplotlib axis object.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Previously initialized axis object
    historical_data : pandas.DataFrame
        Dataframe of historical data
    adm_key : str
        Admin key to use to relate simulation data and geographic areas
    adm_value : int
        Admin code value for area
    col : str
        Column to add to plot
    sim_max_date : pandas.Timestamp
        Latest date in simulation data

    Returns
    -------
    axis : matplotlib.axes.Axes
        Modified axis object with added data
    """
    # Get historical data for area
    actuals = historical_data.loc[historical_data[adm_key] == adm_value]

    if not actuals.empty and col in historical_data.columns:

        actuals = actuals.assign(date=pd.to_datetime(actuals["date"]))
        hist_label = "Historical " + readable_col_names[col]
        actuals.plot(x="date", y=col, ax=axis, color="r", marker="o", ls="", label=hist_label, scaley=False)

        # Set xlim
        axis.set_xlim(actuals["date"].min(), sim_max_date)

    else:
        logger.warning("Historical data missing for " + adm_key + " " + str(adm_value))

    return axis


def add_fit_data_to_plot(axis, fitted_data, adm_key, adm_value, col, sim_max_date):
    """Add historical data for requested column and area to initialized matplotlib axis object.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Previously initialized axis object
    fitted_data : pandas.DataFrame
        Dataframe of fit data
    adm_key : str
        Admin key to use to relate simulation data and geographic areas
    adm_value : int
        Admin code value for area
    col : str
        Column to add to plot
    sim_max_date : pandas.Timestamp
        Latest date in simulation data

    Returns
    -------
    axis : matplotlib.axes.Axes
        Modified axis object with added data
    """

    # TODO: CHANGE STYLE HERE
    # Get historical data for area
    actuals = fitted_data.loc[fitted_data[adm_key] == adm_value]

    if not actuals.empty and col in fitted_data.columns:

        actuals = actuals.assign(date=pd.to_datetime(actuals["date"]))
        fit_label = "Fitted " + readable_col_names[col]
        actuals.plot(x="date", y=col, ax=axis, color="k", marker="", ls="--", label=fit_label, scaley=False)

        # Set xlim
        # axis.set_xlim(actuals["date"].min(), sim_max_date)

    else:
        logger.warning("Fit data missing for " + adm_key + " " + str(adm_value))

    return axis


def preprocess_historical_dates(historical_data, hist_start_date, plot_start, plot_end, min_data_points):
    """Check that historical data has the correct date range and requested number of data points.

    Parameters
    ----------
    historical_data : pandas.DataFrame
        Dataframe with requested historical data
    hist_start_date : str or None
        Plot historical data from this point (formatted as YYYY-MM-DD).
        If None, aligns with simulation start date.
    plot_start : pandas.Timestamp
        Earliest date appearing in simulation data
    plot_end : pandas.Timestamp
        Latest date appearing in simulation data
    min_data_points : int
        Minimum number of historical data points to plot

    Returns
    -------
    historical_data : pandas.DataFrame
        Historical data with the correct date range and number of points
    """
    # Drop data not within requested time range
    historical_data = historical_data.reset_index()
    historical_data = historical_data.assign(date=pd.to_datetime(historical_data["date"]))

    # Use plot start as start date unless otherwise specified
    start_date = plot_start

    if hist_start_date is not None:
        start_date = hist_start_date

    # Check that there are the minimum number of points
    last_hist_date = historical_data["date"].max()
    num_points = (last_hist_date - pd.to_datetime(start_date)).days

    # Shift start date if necessary
    if num_points < min_data_points:
        start_date = last_hist_date - pd.Timedelta(str(min_data_points) + " days")

    historical_data = historical_data.loc[(historical_data["date"] < plot_end) & (historical_data["date"] > start_date)]

    return historical_data


def pool_plot(args):
    """Function to create plots given a tuple of input data.

    This input data should include a pandas DataFrameGroupBy object, the plot
    name, historical data in the same order as the keys in the GroupBy object,
    the columns and quantiles to plot, the admin key, and the directory to
    save plots and CSVs.

    Parameters
    ----------
    args : tuple
        Zipped input data
    """
    (area_code, area_data), name, hist_data, plot_columns, quantiles, adm_key, out_dir, fit_data = args

    # Set date
    area_data = area_data.assign(date=pd.to_datetime(area_data["date"]))

    # Initialize figure
    fig, axs = plt.subplots(
        2,
        sharex=True,
        gridspec_kw={"hspace": 0.1, "height_ratios": [2, 1]},
        figsize=(15, 7),
    )
    fig.suptitle(name.upper())

    # Loop over requested columns
    for i, p_col in enumerate(plot_columns):

        axs[i] = add_col_data_to_plot(axs[i], p_col, area_data, quantiles)

        if fit_data is not None and i <= (len(plot_columns) - 1):
            max_date = area_data["date"].max()
            axs[i] = add_fit_data_to_plot(axs[i], fit_data, adm_key, area_code, p_col, max_date)

        # Plot historical data which is already at the correct level
        if hist_data is not None and i <= (len(plot_columns) - 1):
            max_date = area_data["date"].max()
            axs[i] = add_hist_data_to_plot(axs[i], hist_data, adm_key, area_code, p_col, max_date)

        # Axis styling
        axs[i].grid(True)
        axs[i].legend()
        axs[i].set_ylabel("Count")
        # axs[i].yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

    plot_filename = os.path.join(out_dir, readable_col_names[plot_columns[0]] + "_" + name + ".png")
    plot_filename = plot_filename.replace(" : ", "_")
    plot_filename = plot_filename.replace(" ", "")
    plt.savefig(plot_filename)
    plt.close()

    # Save CSV
    csv_filename = os.path.join(out_dir, name + ".csv")
    area_data.to_csv(csv_filename)


def plot(out_dir, lookup_df, key, sim_data, hist_data, plot_columns, quantiles, num_proc, fit_data):
    """Given a dataframe and a key, creates plots with requested columns.

    For example, a pandas.DataFrame with state-level data would create a plot for
    each unique state. Simulation data is plotted as a line with shaded
    confidence intervals. Historical data is added as scatter points if
    requested.

    Parameters
    ----------
    out_dir : str
        Location to place created plots.
    lookup_df : pandas.DataFrame
        Dataframe containing information relating different geographic
        areas
    key : str
        Key to use to relate simulation data and geographic areas. Must
        appear in lookup and simulation data (and historical data if
        applicable)
    sim_data : pandas.DataFrame
        Simulation data to plot
    hist_data : pandas.DataFrame
        Historical data to add to plot
    plot_columns : list of str
        Columns to plot
    quantiles : list of float, or None
        List of quantiles to plot. If None, will plot all available
        quantiles in data.
    num_proc : int
        Number of processes for multiprocessing
    """
    # If quantiles were not specified, get all quantiles present in data
    if quantiles is None:
        quantiles = sim_data["quantile"].unique()

    # make sure quantiles are sorted
    quantiles.sort()

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

    # Get data in a format we can map for pool
    area_groups = sim_data.groupby(key)
    area_keys = area_groups.groups.keys()

    # Get all names for plots
    all_names = get_all_plot_titles(lookup_df, key, area_keys)

    # Get corresponding historical data
    if hist_data is not None:
        group_hist_data = get_group_historical_data(hist_data, key, area_keys)
    else:
        group_hist_data = [None for _ in range(len(area_keys))]

    # Get all historical fit data
    if fit_data is not None:
        group_fit_data = get_group_historical_data(fit_data, key, area_keys)
    else:
        group_fit_data = [None for _ in range(len(area_keys))]
    # Also pass in adm key, plot columns
    pool_input = zip(
        area_groups,
        all_names,
        group_hist_data,
        [plot_columns for _ in range(len(area_keys))],
        [quantiles for _ in range(len(area_keys))],
        [key for _ in range(len(area_keys))],
        [out_dir for _ in range(len(area_keys))],
        group_fit_data,
    )

    # this tqdm wrapper does the same thing...
    # from tqdm.contrib.concurrent import process_map
    # process_map(pool_plot, pool_input, max_workers=num_proc)
    if (num_proc > 1) or (len(area_groups) == 1):
        with multiprocessing.Pool(num_proc) as p:
            list(tqdm.tqdm(p.imap(pool_plot, pool_input), total=len(area_groups)))
    else:
        for p_inp in tqdm.tqdm(pool_input, total=len(area_groups)):
            pool_plot(p_inp)


def make_plots(
    adm_levels,
    input_directory,
    output_directory,
    lookup_df,
    plot_hist,
    plot_columns,
    quantiles,
    window_size,
    end_date,
    hist_file,
    min_hist_points,
    admin1=None,
    hist_start=None,
    num_proc=16,
    plot_fit=True,
):
    """Wrapper function around plot. Creates plots, aggregating data if necessary.

    Parameters
    ----------
    adm_levels : list of str
        List of ADM levels to make plots for
    input_directory : str
        Location of simulation data
    output_directory : str
        Parent directory to place created plots.
    lookup_df : pandas.DataFrame
        Lookup table for geographic mapping information
    plot_hist : bool
        If True, will plot historical data.
    plot_columns : list of str
        List of columns to plot from data.
    quantiles : list of float, or None
        List of quantiles to plot. If None, will plot all available
        quantiles in data.
    window_size : int
        Size of window (in days) to apply to historical data.
    end_date : str
        Plot data until this date. Must be formatted as YYYY-MM-DD.
        If None, uses last date in simulation.
    hist_file : str
        Path to historical data file. If None, uses either CSSE or
        Covid Tracking data depending on columns requested.
    min_hist_points : int
        Minimum number of historical data points to plot.
    admin1 : list of str, or None
        List of admin1 values to make plots for. If None, a plot will be
        created for every unique admin1 values. Otherwise, plots are
        only made for those requested.
    hist_start : str, or None
        Plot historical data from this point (formatted as YYYY-MM-DD).
        If None, aligns with simulation start date.
    num_proc : int
        Number of processes for multiprocessing
    """
    # Loop over requested levels
    for level in adm_levels:

        filename = level + "_quantiles.csv"

        # Read the requested file from the input dir
        data = pd.read_csv(os.path.join(input_directory, filename))
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
        plot_dir = output_directory
        if level in ["adm2", "adm1"]:

            plot_dir = os.path.join(plot_dir, level.upper())

            # create directory
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

        # For admin2 only: if a admin1 name was passed in, only keep data within that admin1
        if level == "adm2" and admin1 is not None:

            admin2_vals = lookup_df.loc[lookup_df["adm1_name"] == admin1]["adm2"].unique()
            data = data.loc[data["adm2"].isin(admin2_vals)]

        # Read historical data if needed
        hist_data = None
        if plot_hist:

            # Get historical data for requested columns
            hist_data = get_historical_data(plot_columns, level, lookup_df, window_size, hist_file)

            # Check if historical data was not successfully fetched
            if hist_data is None:
                logger.warning("No historical data could be found for: " + str(plot_columns))

            else:
                hist_data = preprocess_historical_dates(hist_data, hist_start, start_date, end_date, min_hist_points)

            # Plot historical data fit line
            if plot_fit:
                fit_data = get_historical_fit(
                    input_directory,
                    plot_columns,
                    level,
                    lookup_df,
                    min_date=hist_data.date.min(),
                )
            else:
                fit_data = None

        plot(
            out_dir=plot_dir,
            lookup_df=lookup_df,
            key=level,
            sim_data=data,
            hist_data=hist_data,
            plot_columns=plot_columns,
            quantiles=quantiles,
            num_proc=num_proc,
            fit_data=fit_data,
        )


def main(cfg):
    """Main entrypoint."""

    logger.info(cfg)

    # Parse arguments
    input_dir = cfg["input_dir"]

    output_dir = input_dir / "plots"  # TODO add optional cli arg
    output_dir.mkdir(exist_ok=True)

    # aggregation levels
    levels = cfg["levels"]

    # Columns for plotting
    plot_cols = cfg["columns"]

    # Make lookup_table from adm mapping
    adm_mapping_file = Path(input_dir) / "metadata" / "adm_mapping.csv"
    adm1_name_map = us.states.mapping("fips", "name")
    adm_mapping_df = pd.read_csv(adm_mapping_file)
    adm_mapping_df["adm1_name"] = adm_mapping_df["adm1"].astype(str).str.zfill(2).map(adm1_name_map)
    adm_mapping_df["adm0_name"] = us.name
    adm_mapping_df["adm0"] = "US"
    lookup_table = adm_mapping_df

    # TODO need to handle/remove all the other olc cli args below
    # Historical data start
    hist_start_date = None  # args.hist_start # TODO deprecate?

    # Parse optional flags
    window = cfg["window_size"]
    plot_historical = True  # TODO make bool flag (it should default to true though): no-hist?

    plot_end_date = None  # make optional arg --end_date
    list_quantiles = None  # make optional arg: --quantiles, -q
    hist_data_file = None  # TODO do we still need this? args.hist_file
    min_hist = cfg["n_hist"]

    # Number of processes for pool
    if cfg["num_proc"] == -1:
        num_proc = multiprocessing.cpu_count() // 2  # todo cli arg
    else:
        num_proc = cfg["num_proc"]

    if num_proc < 1:
        num_proc = 1

    # If a historical file was passed in, make sure hist is also true
    if hist_data_file is not None:
        plot_historical = True

    # Plot
    with matplotlib.style.context("ggplot"), matplotlib.rc_context({"axes.formatter.useoffset": False}):
        make_plots(
            levels,
            input_dir,
            output_dir,
            lookup_table,
            plot_historical,
            plot_cols,
            list_quantiles,
            window,
            plot_end_date,
            hist_data_file,
            min_hist,
            "TODO",  # args.adm1_name,
            hist_start_date,
            num_proc,
        )


if __name__ == "__main__":
    main()
