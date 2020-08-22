import argparse
import datetime
import glob
import logging
import os
from pathlib import Path
import pickle
from datetime import timedelta
from functools import partial
from multiprocessing import Pool, cpu_count

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats

from .util.read_config import bucky_cfg
from .viz.geoid import read_geoid_from_graph, read_lookup


def divide_by_pop(dataframe, cols):
    """Given a dataframe and list of columns, divides the columns by the 
    population column ('N').
    
    Parameters
    ----------
    dataframe : Pandas DataFrame
        Simulation data
    cols : list of strings
        Column names to scale by population

    Returns
    -------
    dataframe : Pandas DataFrame
        Original dataframe with the requested columns scaled

    """
    for col in cols:
        dataframe[col] = dataframe[col] / dataframe["N"]

    return dataframe


# Initialize argument parser
parser = argparse.ArgumentParser(description="Bucky Model postprocessing")

# Required: File to process
parser.add_argument(
    "file",
    default=max(
        glob.glob(bucky_cfg["raw_output_dir"] + "/*/"),
        key=os.path.getctime,
        default="Most recently created folder in raw_output_dir",
    ),
    nargs="?",
    type=str,
    help="File to proess",
)

# Graph file used for this run. Defaults to most recently created
parser.add_argument(
    "-g", "--graph_file", default=None, type=str, help="Graph file used for simulation",
)

# Aggregation levels, e.g. state, county, etc.
parser.add_argument(
    "-l",
    "--levels",
    default=["adm0", "adm1", "adm2"],
    nargs="+",
    type=str,
    help="Levels on which to aggregate",
)

# Quantiles
default_quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
parser.add_argument(
    "-q",
    "--quantiles",
    default=default_quantiles,
    nargs="+",
    type=float,
    help="Quantiles to process",
)

# Top-level output directory
parser.add_argument(
    "-o",
    "--output",
    default=bucky_cfg["output_dir"],
    type=str,
    help="Directory for output files",
)

# Prefix for filenames
parser.add_argument(
    "--prefix",
    default=None,
    type=str,
    help="Prefix for output folder (default is UUID)",
)

# Specify optional end date
parser.add_argument("--end_date", default=None, type=str)

# Can pass in a lookup table to use in place of graph
parser.add_argument(
    "--lookup",
    default=None,
    type=str,
    help="Lookup table defining geoid relationships",
)

# Optional flags
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Print extra information"
)
parser.add_argument(
    "--no_quantiles", action="store_false", help="Skip creating quantiles"
)

if __name__ == "__main__":

    args = parser.parse_args()

    # Start parsing args
    quantiles = args.quantiles
    verbose = args.verbose
    use_quantiles = args.no_quantiles
    prefix = args.prefix

    if verbose:
        logging.info(args)

    # File Management
    top_output_dir = args.output

    # Check if it exists, make if not
    if not os.path.exists(top_output_dir):
        os.makedirs(top_output_dir)

    # Use lookup, add prefix
    if args.lookup is not None:
        lookup_df = read_lookup(args.lookup)
        prefix = Path(args.lookup).stem
    else:
        lookup_df = read_geoid_from_graph(args.graph_file)

    # Create subfolder for this run using UUID of run
    uuid = args.file.split("/")[-2]

    if prefix is not None:
        uuid = prefix + "_" + uuid

    # Create directory if it doesn't exist
    output_dir = os.path.join(top_output_dir, uuid)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get aggregation levels
    agg_levels = args.levels

    lookup_df.set_index("adm2", inplace=True)

    admin2_key = "ADM2_ID"

    # Read H5 file
    run_data = [pd.read_feather(f) for f in glob.glob(args.file + "/*")]
    tot_df = pd.concat(run_data)
    # tot_df = pd.read_hdf(args.file, key='data')

    # Get start date of simulation
    start_date = tot_df["date"].min()

    # If user passes in an end date, use it
    if args.end_date is not None:
        end_date = pd.to_datetime(args.end_date)

    # Otherwise use last day in data
    else:
        end_date = tot_df["date"].max()

    # Drop data not within requested time range
    tot_df = tot_df.loc[(tot_df["date"] <= end_date) & (tot_df["date"] >= start_date)]

    # Some lookups only contain a subset of counties, drop extras if necessary
    unique_adm2 = lookup_df.index
    tot_df = tot_df.loc[tot_df[admin2_key].isin(unique_adm2)]

    # Start reading data
    seird_cols = ["S", "E", "I", "Ic", "Ia", "R", "Rh", "D"]
    # asym = p_df.groupby('rid').mean()['ASYM_FRAC']
    tot_df = tot_df.assign(N=tot_df[seird_cols].sum(axis=1))
    # tot_df = tot_df.merge(asym, left_on='rid', right_index=True)

    # For cases, don't include asymptomatic
    infected_columns = ["I", "Ic"]  # , 'Ia']
    tot_df = tot_df.assign(cases_active=tot_df[infected_columns].sum(axis=1))

    # Calculate hospitalization
    tot_df = tot_df.assign(hospitalizations=tot_df[["Rh", "Ic"]].sum(axis=1))
    # tot_df = tot_df.assign(hospitalizations=tot_df['Rh'])

    # Drop columns other than these
    keep_cols = [
        "N",
        "hospitalizations",
        "NCR",
        "CCR",
        "NC",
        "CC",
        "ND",
        "D",
        "Ia",
        "ICU",
        "VENT",
        "cases_active",
        admin2_key,
        "date",
        "rid",
        "CASE_REPORT",
        "Reff",
        "doubling_t",
    ]

    tot_df = tot_df.drop(columns=[c for c in tot_df.columns if c not in keep_cols])

    # Rename
    tot_df.rename(
        columns={
            "D": "cumulative_deaths",
            "NC": "daily_cases",
            "ND": "daily_deaths",
            "CC": "cumulative_cases",
            "Ia": "cases_asymptomatic_active",
            "NCR": "daily_cases_reported",
            "CCR": "cumulative_cases_reported",
        },
        inplace=True,
    )

    # Multiply column by N, then at end divide by aggregated N
    pop_mean_cols = ["CASE_REPORT", "Reff", "doubling_t"]
    for col in pop_mean_cols:
        tot_df[col] = tot_df[col] * tot_df["N"]

    # No columns should contain negatives or NaNs
    nan_vals = tot_df.isna().sum()
    if nan_vals.sum() > 0:
        logging.error("NANs are present in output data: \n" + str(nan_vals))
        # raise ValueError('NANs are present in output data: ' + str(nan_vals))

    # Check all columns except date for negative values
    if (tot_df.drop(columns=["date"]).lt(-1).sum() > 0).any():
        logging.error("Negative values are present in output data.")
        # raise ValueError('Negative values are present in output data: ' + str(negative_vals))

    # Check for floating point errors
    if (tot_df.drop(columns=["date"]).lt(0).sum() > 0).any():
        logging.warning("Floating point errors are present in output data.")

    for level in agg_levels:

        logging.info("Currently processing: " + level)

        if level != "adm2":

            # Create a mapping for aggregation level
            level_dict = lookup_df[level].to_dict()
            levels = np.unique(list(level_dict.values()))
        else:
            levels = lookup_df.index.values
            level_dict = {x: x for x in levels}

        level_map = dict(enumerate(levels))
        level_inv_map = {v: k for k, v in level_map.items()}

        # Apply map
        tot_df[level] = (
            tot_df[admin2_key].map(level_dict).map(level_inv_map).astype(int)
        )

        # Compute quantiles
        if use_quantiles:
            with Pool(cpu_count()) as p:
                q_df = (
                    pd.concat(
                        p.map(
                            partial(
                                pd.DataFrame.quantile, q=quantiles, numeric_only=False
                            ),
                            [
                                g.assign(level=n[0], date=n[1])
                                for n, g in tot_df.groupby([level, "date", "rid"])
                                .sum()
                                .groupby([level, "date"])
                            ],
                        )
                    )
                    .reset_index()
                    .rename(columns={"index": "q", "level": level})
                )
            q_df[level] = q_df[level].round().astype(int).map(level_map)
            q_df.set_index([level, "date", "q"], inplace=True)
            q_df = q_df.assign(
                cases_per_100k=(q_df["cases_active"] / q_df["N"]) * 100000.0
            )
            q_df = divide_by_pop(q_df, pop_mean_cols)

        # Compute mean
        mean_df = (
            tot_df.groupby([level, "date", "rid"])
            .sum()
            .groupby([level, "date"])
            .mean()
            .assign(q=-1)
            .reset_index()
        )
        mean_df[level] = mean_df[level].map(level_map)
        mean_df.set_index([level, "date", "q"], inplace=True)
        mean_df.index.set_names([level, "date", "q"], inplace=True)
        mean_df = mean_df.assign(
            cases_per_100k=(mean_df["cases_active"] / mean_df["N"]) * 100000.0
        )
        mean_df = divide_by_pop(mean_df, pop_mean_cols)

        # Compute standard deviation
        std_df = (
            tot_df.groupby([level, "date", "rid"])
            .sum()
            .groupby([level, "date"])
            .std()
            .assign(q=-2)
            .reset_index()
        )
        std_df[level] = std_df[level].map(level_map)
        std_df.set_index([level, "date", "q"], inplace=True)
        std_df.index.set_names([level, "date", "q"], inplace=True)

        # Get mean_df's N to compute cases_per_100k
        tmp_N = (
            mean_df.reset_index()
            .drop(columns=["q"])
            .assign(q=-2)
            .set_index(std_df.index.names)
        )
        std_df = std_df.assign(
            cases_per_100k=(std_df["cases_active"] / tmp_N["N"]) * 100000.0
        )

        for col in pop_mean_cols:
            std_df[col] = std_df[col] / tmp_N["N"]

        # Column management
        if level != admin2_key:

            # Remove unncessary FIPS column from new dataframe
            if use_quantiles:
                q_df.drop(columns=admin2_key, inplace=True)
            mean_df.drop(columns=admin2_key, inplace=True)
            std_df.drop(columns=admin2_key, inplace=True)

            # And remove now unnecessary level column from original dataframe
            tot_df.drop(columns=[level], inplace=True)
        else:
            tot_df[level] = tot_df[admin2_key].map(level_map)

        if verbose:
            logging.info("Mean dataframe:")
            logging.info(mean_df.head())

            logging.info("\nStandard deviation dataframe:")
            logging.info(std_df.head())

            if use_quantiles:
                logging.info("\nQuantiles dataframe:")
                logging.info(q_df.head())

        # Create two output files: one for quantiles and one for mean+standard deviation
        if use_quantiles:
            q_df.to_csv(os.path.join(output_dir, level + "_quantiles.csv"), header=True)

        # Combine mean and standard deviation into one DF
        mean_df = pd.concat([mean_df, std_df])
        mean_df = mean_df.reset_index()

        # Rename q column
        mean_df = mean_df.assign(stat=mean_df["q"].map({-1: "mean", -2: "std"}))
        mean_df.drop(columns=["q"], inplace=True)
        mean_df.set_index([level, "date", "stat"], inplace=True)
        mean_df.to_csv(os.path.join(output_dir, level + "_mean_std.csv"), header=True)
