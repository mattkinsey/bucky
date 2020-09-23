import argparse
import datetime
import glob
import logging
import os
from pathlib import Path
import pickle
from datetime import timedelta
from functools import partial
from multiprocessing import Pool, cpu_count, Queue, current_process, RLock, Process
import gc

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm

from .util.read_config import bucky_cfg
from .viz.geoid import read_geoid_from_graph, read_lookup

import cupy as cp

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
default_quantiles = [0.01, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.7, 0.750, 0.800, 0.85, 0.9, 0.950, 0.975, 0.990]
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

parser.add_argument('-n', '--nprocs', default=1, type=int)

parser.add_argument('--verify', action='store_true', help='Verify the quality of the data')

parser.add_argument('--no-sort', '--no_sort', action='store_true', help='Skip sorting the aggregated files')

# Optional flags
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Print extra information"
)

if __name__ == "__main__":

    args = parser.parse_args()

    #logging.basicConfig(level=logging.INFO)

    # Start parsing args
    quantiles = args.quantiles
    verbose = args.verbose
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
        if prefix is None:
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

    write_header = True    
    all_files = glob.glob(args.file + '/*')
    all_files_df = pd.DataFrame([x.split('/')[-1].split('.')[0].split('_') for x in all_files], columns=['rid', 'date'])
    dates = all_files_df.date.unique().tolist()

    to_write = Queue()

    def writer():
        # Call to_write.get() until it returns None
        has_header_dict = {}
        for fname, df in iter(to_write.get, None):
            if fname in has_header_dict:
                df.to_csv(fname, header=False, mode='a')
            else: 
                df.to_csv(fname, header=True, mode='w')
                has_header_dict[fname] = True
    write_thread = Process(target=writer)
    write_thread.start()


    def process_date(date, write_header=False, write_queue=to_write):
        date_files = glob.glob(args.file + "/*_" + str(date) + '.feather')#[:NFILES]

        # Read feather files
        run_data = [pd.read_feather(f) for f in date_files]
        tot_df = pd.concat(run_data)

        # force GC to free up lingering cuda allocs
        del run_data
        gc.collect()
    
        # Get start date of simulation
        #start_date = tot_df["date"].min()
    
        # If user passes in an end date, use it
        if args.end_date is not None:
            end_date = pd.to_datetime(args.end_date)
    
        # Otherwise use last day in data
        else:
            end_date = tot_df["date"].max()
    
            # Drop data not within requested time range
            tot_df = tot_df.loc[(tot_df["date"] <= end_date)]
    
        # Some lookups only contain a subset of counties, drop extras if necessary
        unique_adm2 = lookup_df.index
        tot_df = tot_df.loc[tot_df[admin2_key].isin(unique_adm2)]
    
        # Start reading data
        seird_cols = ["S", "E", "I", "Ic", "Ia", "R", "Rh", "D"]
        N=tot_df[seird_cols].sum(axis=1)
    
        # For cases, don't include asymptomatic
        infected_columns = ["I", "Ic"]  # , 'Ia']
        cases_active=tot_df[infected_columns].sum(axis=1)
        # Calculate hospitalization
        hospitalizations=tot_df[["Rh", "Ic"]].sum(axis=1)
        
        tot_df = tot_df.assign(N=N, cases_active=cases_active, hospitalizations=hospitalizations)
    
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
            "nCCR",
            "nNCR",
            "NH",
        ]
    
        tot_df = tot_df[keep_cols]
        gc.collect()
    
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
    
        if args.verify:
            # Check all columns except date for negative values
            if (tot_df.drop(columns=["date"]).lt(-1).sum() > 0).any(): # TODO this drop does a deep copy and is super slow
                logging.error("Negative values are present in output data.")
    
            # Check for floating point errors
            if (tot_df.drop(columns=["date"]).lt(0).sum() > 0).any(): # TODO same here
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
            # TODO why is this in the for loop? pretty sure we can move it but check for deps
            def quantiles_group(tot_df):
                tot_df_stacked = tot_df.stack()
                tot_df_unstack = tot_df_stacked.unstack('rid')
                percentiles = cp.array(quantiles)*100.
                test = cp.percentile(cp.array(tot_df_unstack.to_numpy()),q=percentiles,axis=1)
                q_df = pd.DataFrame(test.T.get(), index=tot_df_unstack.index, columns=quantiles).unstack().stack(level=0).reset_index().rename(columns={'level_2': 'q'})
                return q_df

            g = tot_df.groupby([level,'date','rid']).sum().groupby(level)

            q_df = g.apply(quantiles_group)

            q_df[level] = q_df[level].round().astype(int).map(level_map)
            q_df.set_index([level, "date", "q"], inplace=True)
            q_df = q_df.assign(
                cases_per_100k=(q_df["cases_active"] / q_df["N"]) * 100000.0
            )
            q_df = divide_by_pop(q_df, pop_mean_cols)
    
            # Column management
            if level != admin2_key:
                del q_df[admin2_key]
    
            if verbose:
                logging.info("\nQuantiles dataframe:")
                logging.info(q_df.head())

            # Push output df to write queue
            write_queue.put((os.path.join(output_dir, level + "_quantiles.csv"),q_df))

    
    
    with Pool(processes=args.nprocs) as pool:
        for _ in tqdm(pool.imap_unordered(process_date, dates), total=len(dates)):
            pass

    to_write.put(None)
    to_write.close()
    #to_write.join_thread()
    #write_thread.close()
    #write_thread.join()


    #sort output csvs
    if not args.no_sort:
        for level in args.levels:
            fname = os.path.join(output_dir, level + "_quantiles.csv")
            print('Sorting output file ' + fname, end='... ')
            df = pd.read_csv(fname)
            df.sort_values([level, 'date', 'q'], inplace=True)
            df.to_csv(fname, index=False)
            print(' Done')

