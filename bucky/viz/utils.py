"""Utility functions for plotting."""
from pathlib import Path

import numpy as np
import pandas as pd
import us
from loguru import logger

from ..data.timeseries import BuckyFittedCaseData, CSSEData, HHSData
from ..numerical_libs import sync_numerical_libs, xp
from ..util.array_utils import rolling_window


def get_simulation_data(input_dir, level):
    """Gets Bucky simulation data from the requested input directory at the requested
    admin level.

    Parameters
    ----------
    input_dir : matplotlib.axes.Axes
        Directory in which Bucky simulation data is located
    level : str
        Requested admin level

    Returns
    -------
    sim_data : pandas.DataFrame
        Bucky simulation data in DataFrame
    """

    # Read file
    filename = input_dir / (level + "_quantiles.csv")
    sim_data = pd.read_csv(filename)  # TODO specify engine for speed c engine
    sim_data = sim_data.assign(date=pd.to_datetime(sim_data["date"]))

    return sim_data


@sync_numerical_libs
def get_fitted_data(cfg, level, date_range):
    """Gets Bucky fitted historical data for the requested columns and dates.
    admin level.

    Parameters
    ----------
    cfg : BuckyConfig
        BuckyConfig object with various plot-related parameters
    level : str
        Requested admin level
    date_range : tuple
        Date range for which to get fitted data

    Returns
    -------
    df : pandas.DataFrame
        Bucky fitted historical data in DataFrame
    """

    # Requested column(s) determine file to open
    case_cols = ["daily_reported_cases", "daily_deaths", "cumulative_cases", "cumulative_deaths"]
    hosp_cols = ["daily_hospitalizations", "current_hospitalizations"]

    df = None
    # TODO remove when admin mapping modified
    level_int = int(level[-1])
    level_mapping = cfg["adm_mapping"].mapping("ids", "abbrs", level=level_int)
    # Get data for each requested column
    for col in cfg["columns"]:

        # Determine whether case or hosp data

        # Case data
        if col in case_cols:
            filename = cfg["input_dir"] / "metadata" / "csse_fitted_timeseries.csv"
            data = BuckyFittedCaseData.from_csv(filename, valid_date_range=date_range, adm_mapping=cfg["adm_mapping"])

            # Aggregate adm data if necessary
            if int(level[-1]) != data.adm_level:
                data = data.sum_adm_level(level=level)

            # Get requested column
            if col == "daily_reported_cases":
                col_data = data.incident_cases
            else:
                col_data = data.incident_deaths

        elif col in hosp_cols:
            filename = cfg["input_dir"] / "metadata" / "hhs_fitted_timeseries.csv"
            data = HHSData.from_csv(filename, valid_date_range=date_range, adm_mapping=cfg["adm_mapping"])

            # Aggregate to requested level if required
            if int(level[-1]) != data.adm_level:
                data = data.sum_adm_level(level=level)

            # Get requested column
            if "daily" in col:
                col_data = data.incident_hospitalizations
            else:
                col_data = data.current_hospitalizations

        else:
            logger.error("Requested column is not available in fitted data.")
            raise NotImplementedError

        # Place into dataframe
        col_df = pd.DataFrame(xp.to_cpu(col_data), columns=xp.to_cpu(data.adm_ids), index=xp.to_cpu(data.dates))
        col_df = col_df.unstack()
        col_df.index = col_df.index.set_names([level, "date"])
        col_df.name = col

        if df is None:
            df = pd.DataFrame(col_df)
        else:
            df = pd.concat([df, pd.DataFrame(col_df)], axis=1)

    # Reset index
    df = df.reset_index()

    # Add string names for admin levels
    df[level + "_name"] = df[level].map(level_mapping)

    # Simulation data for adm0 is keyed on "US" instead of id
    if level == "adm0":
        df[level] = df[level].map(level_mapping)

    return df


@sync_numerical_libs
def get_historical_data(cfg, level, date_range):
    """Gets historical data for the requested columns and dates.
    admin level.

    Parameters
    ----------
    cfg : BuckyConfig
        BuckyConfig object with various plot-related parameters
    level : str
        Requested admin level
    date_range : tuple
        Date range for which to get fitted data

    Returns
    -------
    df : pandas.DataFrame
        Historical data in DataFrame
    """

    df = None

    # TODO remove when admin mapping modified
    level_int = int(level[-1])
    level_mapping = cfg["adm_mapping"].mapping("ids", "abbrs", level=level_int)

    # Get data for each requested column
    for col in cfg["columns"]:

        # CSSE Data
        if col in ["daily_reported_cases", "daily_deaths"]:

            filename = cfg["hist_data_dir"] / "csse_timeseries.csv"
            data = CSSEData.from_csv(filename, valid_date_range=date_range, adm_mapping=cfg["adm_mapping"])

            # Aggregate adm data if necessary
            if int(level[-1]) != data.adm_level:
                data = data.sum_adm_level(level=level)

            # Get requested column
            if col == "daily_reported_cases":
                col_data = data.incident_cases
            else:
                col_data = data.incident_deaths

        # HHS hospitalization data
        elif col in ["daily_hospitalizations", "current_hospitalizations"]:
            filename = cfg["hist_data_dir"] / "hhs_timeseries.csv"
            data = HHSData.from_csv(filename, valid_date_range=date_range, adm_mapping=cfg["adm_mapping"])

            # Aggregate to requested level if required
            if int(level[-1]) != data.adm_level:
                data = data.sum_adm_level(level=level)

            # Get requested column
            if "daily" in col:
                col_data = data.incident_hospitalizations
            else:
                col_data = data.current_hospitalizations

        else:
            logger.error("Requested column is not available in historical data.")
            raise NotImplementedError

        # Window if requested
        if cfg["window_size"] > 1:
            col_data = xp.mean(rolling_window(col_data, cfg["window_size"]), axis=-1)

        # Place into dataframe
        col_df = pd.DataFrame(xp.to_cpu(col_data), columns=xp.to_cpu(data.adm_ids), index=xp.to_cpu(data.dates))
        col_df = col_df.unstack()
        col_df.index = col_df.index.set_names([level, "date"])
        col_df.name = col

        if df is None:
            df = pd.DataFrame(col_df)
        else:
            df = pd.concat([df, pd.DataFrame(col_df)], axis=1)

    # Reset index
    df = df.reset_index()

    # Add string names for admin levels
    df[level + "_name"] = df[level].map(level_mapping)

    # Simulation data for adm0 is keyed on "US" instead of id
    if level == "adm0":
        df[level] = df[level].map(level_mapping)

    return df
