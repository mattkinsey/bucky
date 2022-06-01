"""Creates plots for Bucky data."""
import datetime
import multiprocessing
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import tqdm
import us
from loguru import logger

from ..data.adm_mapping import AdminLevelMapping
from ..util.util import _banner
from .readable_col_names import readable_col_names
from .utils import get_fitted_data, get_historical_data, get_simulation_data


def add_simulation_data_to_axis(axis, col, sim_data, quantiles):
    """Adds simulation data for requested column to initialized matplotlib axis object.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Previously initialized axis object
    col : str
        Column to add to plot
    sim_data : pandas.DataFrame
        Area simulation data
    quantiles : list of float, or None
        List of quantiles to plot. If None, will plot all available
        quantiles in data.

    Returns
    -------
    axis : matplotlib.axes.Axes
        Modified axis object with added data
    """

    # Index by date and quantile
    sim_data = sim_data.set_index(["date", "quantile"]).sort_index()

    # Middle is median
    num_quantiles = len(quantiles)
    median_data = sim_data.xs(quantiles[int(num_quantiles / 2)], level=1)[col]
    dates = median_data.index.values

    # Plot median
    median_data.plot(
        linewidth=1.5,
        color="k",
        alpha=0.75,
        label=readable_col_names[col],
        ax=axis,
    )

    # Iterate over pairs of quantiles
    # Scale opacity
    alpha = 1.0 / (num_quantiles // 2)
    for q in range(num_quantiles // 2):

        lower_q = quantiles[q]
        upper_q = quantiles[num_quantiles - 1 - q]

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


def scale_y_axis(axis, col, sim_data, quantiles, hist_data, fit_data, zero_as_ymin=True):
    """Adds simulation data for requested column to initialized matplotlib axis object.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Previously initialized axis object
    col : str
        Column to add to plot
    sim_data : pandas.DataFrame
        Area simulation data
    quantiles : list of float, or None
        List of quantiles in data
    hist_data : pandas.DataFrame
        Historical data
    fit_data : pandas.DataFrame
        Fitted historical data
    zero_as_ymin: bool
        If true, uses zero as y-axis minimum
    Returns
    -------
    axis : matplotlib.axes.Axes
        Modified axis object with modified y-axis limits
    """

    limits = []

    # Get min/max of median
    sim_data = sim_data.set_index(["date", "quantile"])
    num_quantiles = len(quantiles)
    median_data = sim_data.xs(quantiles[int(num_quantiles / 2)], level=1)[col]
    limits.append([median_data.min(), median_data.max()])

    # After 30 days, need 0.25-0.75 quantiles shown
    sim_data = sim_data.reset_index()

    # Only applies if run is longer than 30 days
    if len(sim_data["date"].unique()) > 30:
        quantile_mask = (sim_data["quantile"] >= 0.25) & (sim_data["quantile"] <= 0.75)
        sel_sim_data = sim_data.loc[quantile_mask]
        sel_sim_data = sel_sim_data.loc[sel_sim_data["date"] > sim_data["date"].min() + pd.DateOffset(29)]
        limits.append([sel_sim_data[col].min(), sel_sim_data[col].max()])

    # Last 7 days of historical data should be displayed
    if len(hist_data["date"].unique()) <= 7:
        hist_date_max = hist_data["date"].max()
        hist_date_min = hist_data["date"].min()
    else:
        hist_date_max = hist_data["date"].max()
        hist_date_min = hist_data["date"].max() - pd.DateOffset(7)

    date_mask = (hist_data["date"] > hist_date_min) & (hist_data["date"] <= hist_date_max)
    last_hist_week = hist_data.loc[date_mask]
    limits.append([last_hist_week[col].min(), hist_data[col].max()])

    # Get min and max of current limits
    limits = np.array(limits)
    y_min = np.min(limits[:, 0])
    y_max = np.max(limits[:, 1])

    if zero_as_ymin:
        y_min = 0.0

    axis.set_ylim(y_min, y_max)

    return axis


def _pool_plot(args):
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

    (area_code, area_data), (_, hist_data), (_, fit_data), level, level_mapping, cols, quantiles, output_dir = args

    # Set date
    area_data = area_data.assign(date=pd.to_datetime(area_data["date"]))

    # Get plot name
    # TODO fix with new admin mapping
    if level == "adm0":
        area_name = area_data[level].unique()[0]
    else:
        area_name = level_mapping[area_code]

    # Initalize figure with as many subplots as columns
    num_subplots = len(cols)

    # Make all evenly sized for now
    # TODO - Make this a parameter?
    ratios = [1.0 for subplot in range(num_subplots)]

    fig, axs = plt.subplots(
        num_subplots,
        sharex=True,
        gridspec_kw={"hspace": 0.1, "height_ratios": ratios},
        figsize=(15, 7),
    )

    fig.suptitle(area_name.upper())

    # Loop over requested columns
    for i, col in enumerate(cols):

        # Get axis for this column
        if num_subplots > 1:
            axis = axs[i]
        else:
            axis = axs

        # Add Bucky data to the axis
        axis = add_simulation_data_to_axis(axis, col, area_data, quantiles)

        # Add historical data (optional)
        if hist_data is not None:
            hist_data = hist_data.assign(date=pd.to_datetime(hist_data["date"]))

            # Check this is not empty
            if not hist_data.empty and col in hist_data.columns:
                hist_label = "Historical " + readable_col_names[col]
                hist_data.plot(x="date", y=col, ax=axis, color="r", marker="o", ls="", label=hist_label, scaley=False)
            else:
                logger.warning(level + " historical data missing for " + area_name)

        # Add fitted data (optional)
        if fit_data is not None:
            fit_data = fit_data.assign(date=pd.to_datetime(fit_data["date"]))

            # Check this is not empty
            if not fit_data.empty and col in fit_data.columns:
                fit_label = "Fitted " + readable_col_names[col]
                fit_data.plot(x="date", y=col, ax=axis, color="k", marker="", ls="--", label=fit_label, scaley=False)
            else:
                logger.warning(level + " fitted data missing for " + area_name)

        axis = scale_y_axis(axis, col, area_data, quantiles, hist_data, fit_data)

        # Axis styling
        axis.grid(True)
        axis.legend()
        axis.set_ylabel("Count")

    plot_filename = os.path.join(output_dir, readable_col_names[cols[0]] + "_" + area_name + ".png")
    plot_filename = plot_filename.replace(" : ", "_")
    plot_filename = plot_filename.replace(" ", "")
    plt.savefig(plot_filename)
    plt.close()

    # Save CSV
    csv_filename = os.path.join(output_dir, area_name + ".csv")
    area_data.to_csv(csv_filename, index=False)


def default_plot(cfg):
    """Creates default Bucky plot, one plot per unique ADM region containing as many subplots as columns requested.

    Parameters
    ----------
    cfg : BuckyConfig
        BuckyConfig object with various plot-related parameters
    """

    cols = list(cfg["columns"])
    output_dir = cfg["output_dir"]
    admin_mapping = cfg["adm_mapping"]

    # Loop over levels
    for level in cfg["levels"]:

        # Get plot directory
        if level != "adm0":
            plot_dir = output_dir / level.upper()
        else:
            plot_dir = output_dir

        # Get data
        sim_data = get_simulation_data(cfg["input_dir"], level)

        # Read dates
        max_hist_date = sim_data["date"].max().to_pydatetime()
        min_hist_date = sim_data["date"].min().to_pydatetime() - datetime.timedelta(days=cfg["n_hist"] - 1)

        # Get all quantiles and make sure they are sorted
        quantiles = sim_data["quantile"].unique()
        quantiles.sort()

        # Group by adm key
        group_sim_data = sim_data.groupby(level)
        num_admin_groups = len(group_sim_data.groups.keys())

        # Historical data (optional)
        group_hist_data = [(None, None) for _ in range(num_admin_groups)]
        if cfg["plot_hist"]:
            hist_data = get_historical_data(cfg, level, date_range=(min_hist_date, max_hist_date))

            # Group by adm key
            group_hist_data = hist_data.groupby(level)

        # Fitted historical data (optional)
        group_fit_data = [(None, None) for _ in range(num_admin_groups)]
        if cfg["plot_fit"]:
            fit_data = get_fitted_data(cfg, level, date_range=(min_hist_date, max_hist_date))

            # Group by adm key
            group_fit_data = fit_data.groupby(level)

        # TODO remove when admin mapping modified
        level_int = int(level[-1])
        level_mapping = admin_mapping.mapping("ids", "abbrs", level=level_int)

        # Check that all groups are in the same order before sending to pool
        if cfg["plot_hist"]:

            all_adm_vals_match = True

            # Get groups
            sim_groupkeys = list(group_sim_data.groups.keys())
            hist_groupkeys = list(group_hist_data.groups.keys())

            if sim_groupkeys != hist_groupkeys:
                all_adm_vals_match = False

            # Also compare with fit data (optional)
            if cfg["plot_fit"]:

                fit_groupkeys = list(group_fit_data.groups.keys())

                if sim_groupkeys != fit_groupkeys or hist_groupkeys != fit_groupkeys:
                    all_adm_vals_match = False

            if not all_adm_vals_match:
                logger.error("Mismatch between adm groups in simulation and historical data")
                raise ValueError

        # Data structure for pool
        pool_input = zip(
            group_sim_data,
            group_hist_data,
            group_fit_data,
            [level for _ in range(num_admin_groups)],
            [level_mapping for _ in range(num_admin_groups)],
            [cols for _ in range(num_admin_groups)],
            [quantiles for _ in range(num_admin_groups)],
            [plot_dir for _ in range(num_admin_groups)],
        )

        num_proc = cfg["num_proc"]
        if (num_proc > 1) or (num_admin_groups != 1):
            with multiprocessing.Pool(num_proc) as p:
                list(tqdm.tqdm(p.imap(_pool_plot, pool_input), total=num_admin_groups))
        else:
            for p_inp in tqdm.tqdm(pool_input, total=num_admin_groups):
                _pool_plot(p_inp)


def main(cfg):
    """Main entrypoint."""
    _banner("Generating timeseries plots with quantiles")
    logger.debug(cfg)

    # Parse config
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]

    # Check if output dir was passed in
    if output_dir is None:
        output_dir = input_dir / "plots"
        output_dir.mkdir(exist_ok=True)
        cfg["output_dir"] = output_dir

    # aggregation levels
    levels = cfg["levels"]

    # Add subfolder for adms other than adm0
    for level in levels:
        if level != "adm0":
            adm_output_dir = output_dir / level.upper()
            adm_output_dir.mkdir(exist_ok=True)

    # Check if n_hist is 0
    if cfg["n_hist"] < 1:
        cfg["plot_fit"] = False

    # Read admin mapping object
    cfg["adm_mapping"] = AdminLevelMapping.from_csv(cfg["adm_mapping_file"])

    default_plot(cfg)


if __name__ == "__main__":
    main()
