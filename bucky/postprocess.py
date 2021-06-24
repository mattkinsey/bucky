"""Postprocesses data across dates and simulation runs before aggregating at geographic levels (ADM0, ADM1, or ADM2)."""
import argparse
import gc
import glob
import importlib
import logging
import os
import queue
import sys
import threading
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.dataset as ds
import pyarrow.types as pat
import tqdm

from .numerical_libs import enable_cupy, reimport_numerical_libs, xp
from .util.read_config import bucky_cfg
from .viz.geoid import read_lookup

# supress pandas warning caused by pyarrow and the cupy asyncmempool
warnings.simplefilter(action="ignore", category=FutureWarning)

cupy_found = importlib.util.find_spec("cupy") is not None

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
    "-g",
    "--graph_file",
    default=None,
    type=str,
    help="Graph file used for simulation",
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
default_quantiles = [
    0.01,
    0.025,
    0.050,
    0.100,
    0.150,
    0.200,
    0.250,
    0.300,
    0.350,
    0.400,
    0.450,
    0.500,
    0.550,
    0.600,
    0.650,
    0.7,
    0.750,
    0.800,
    0.85,
    0.9,
    0.950,
    0.975,
    0.990,
]
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


parser.add_argument("-gpu", "--gpu", action="store_true", default=cupy_found, help="Use cupy instead of numpy")

parser.add_argument("--verify", action="store_true", help="Verify the quality of the data")


# Optional flags
parser.add_argument("-v", "--verbose", action="store_true", help="Print extra information")


# TODO move this to util
def pinned_array(array):
    """Allocate a cudy pinned array that shares mem with an input numpy array."""
    # first constructing pinned memory
    mem = xp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src


def main(args=None):
    """Main method for postprocessing the raw outputs from an MC run."""
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args()

    # Start parsing args
    quantiles = args.quantiles
    verbose = args.verbose
    prefix = args.prefix
    use_gpu = args.gpu

    if verbose:
        logging.info(args)

    # File Management
    top_output_dir = args.output

    # Check if it exists, make if not
    if not os.path.exists(top_output_dir):
        os.makedirs(top_output_dir)

    # Use lookup, add prefix
    # TODO need to handle lookup weights
    if args.lookup is not None:
        lookup_df = read_lookup(args.lookup)
        if prefix is None:
            prefix = Path(args.lookup).stem
    # TODO if args.lookup we need to check it for weights

    # Create subfolder for this run using UUID of run
    uuid = args.file.split("/")[-2]

    if prefix is not None:
        uuid = prefix + "_" + uuid

    # Create directory if it doesn't exist
    output_dir = os.path.join(top_output_dir, uuid)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_dir = os.path.join(args.file, "data/")
    metadata_dir = os.path.join(args.file, "metadata/")

    adm_mapping = pd.read_csv(os.path.join(metadata_dir, "adm_mapping.csv"))
    dates = pd.read_csv(os.path.join(metadata_dir, "dates.csv"))
    dates = dates["date"].to_numpy()

    n_adm2 = len(adm_mapping)
    adm2_sorted_ind = xp.argsort(xp.array(adm_mapping["adm2"].to_numpy()))

    if use_gpu:
        enable_cupy(optimize=True)
        reimport_numerical_libs("postprocess")

    per_capita_cols = [
        "cumulative_reported_cases",
        "cumulative_deaths",
        "current_hospitalizations",
        "daily_reported_cases",
        "daily_deaths",
    ]
    pop_weighted_cols = [
        "case_reporting_rate",
        "R_eff",
    ]

    adm_mapping["adm0"] = 1
    adm_map = adm_mapping.to_dict(orient="list")
    adm_map = {k: xp.array(v)[adm2_sorted_ind] for k, v in adm_map.items()}
    adm_array_map = {k: xp.unique(v, return_inverse=True)[1] for k, v in adm_map.items()}
    adm_sizes = {k: xp.to_cpu(xp.max(v) + 1).item() for k, v in adm_array_map.items()}
    adm_level_values = {k: xp.to_cpu(xp.unique(v)) for k, v in adm_map.items()}
    adm_level_values["adm0"] = np.array(["US"])

    if args.lookup is not None and "weight" in lookup_df.columns:
        weight_series = lookup_df.set_index("adm2")["weight"].reindex(adm_mapping["adm2"], fill_value=0.0)
        weights = np.array(weight_series.to_numpy(), dtype=np.float32)
        # TODO we should ignore all the adm2 not in weights rather than just 0ing them (it'll go alot faster)
    else:
        weights = np.ones_like(adm2_sorted_ind, dtype=np.float32)

    write_queue = queue.Queue()

    def _writer():
        """Write thread that will pull from a queue."""
        # Call to_write.get() until it returns None
        file_tables = {}
        for fname, q_dict in iter(write_queue.get, None):
            df = pd.DataFrame(q_dict)
            id_col = df.columns[df.columns.str.contains("adm.")].values[0]
            df = df.set_index([id_col, "date", "quantile"])
            df = df.reindex(sorted(df.columns), axis=1)
            if fname in file_tables:
                tmp = pa.table(q_dict)
                file_tables[fname] = pa.concat_tables([file_tables[fname], tmp])
            else:
                file_tables[fname] = pa.table(q_dict)
            write_queue.task_done()

        # dump tables to disk
        for fname in tqdm.tqdm(file_tables):
            df = file_tables[fname].to_pandas()
            id_col = df.columns[df.columns.str.contains("adm.")].values[0]
            df = df.set_index([id_col, "date", "quantile"])
            df = df.reindex(sorted(df.columns), axis=1)
            df.to_csv(fname, header=True, mode="w")
        write_queue.task_done()

    write_thread = threading.Thread(target=_writer)
    write_thread.start()

    # TODO this depends on out of scope vars, need to clean that up
    def pa_array_quantiles(array, level):
        """Calculate the quantiles of a pyarrow array after shipping it to the GPU."""
        data = array.to_numpy().reshape(-1, n_adm2)
        data = data[:, adm2_sorted_ind]

        data_gpu = xp.array(data.T)

        if adm_sizes[level] == 1:
            # TODO need switching here b/c cupy handles xp.percentile weird with a size 1 dim :(
            if use_gpu:
                level_data_gpu = xp.sum(data_gpu, axis=0)  # need this if cupy
            else:
                level_data_gpu = xp.sum(data_gpu, axis=0, keepdims=True).T  # for numpy
            q_data_gpu = xp.empty((len(percentiles), adm_sizes[level]), dtype=level_data_gpu.dtype)
            # It appears theres a cupy bug when the 1st axis of the array passed to percentiles has size 1
            xp.percentile(level_data_gpu, q=percentiles, axis=0, out=q_data_gpu)
        else:
            level_data_gpu = xp.zeros((adm_sizes[level], data_gpu.shape[1]), dtype=data_gpu.dtype)
            xp.scatter_add(level_data_gpu, adm_array_map[level], data_gpu)
            q_data_gpu = xp.empty((len(percentiles), adm_sizes[level]), dtype=level_data_gpu.dtype)
            xp.percentile(level_data_gpu, q=percentiles, axis=1, out=q_data_gpu)
        return q_data_gpu

    percentiles = xp.array(quantiles, dtype=np.float64) * 100.0
    quantiles = np.array(quantiles)
    for date_i, date in enumerate(tqdm.tqdm(dates)):
        dataset = ds.dataset(data_dir, format="parquet", partitioning=["date"])
        table = dataset.to_table(filter=ds.field("date") == "date=" + str(date_i))
        table = table.drop(("date", "rid", "adm2_id"))  # we don't need these b/c metadata
        pop_weight_table = table.select(pop_weighted_cols)
        table = table.drop(pop_weighted_cols)

        w = np.ravel(np.broadcast_to(weights, (table.shape[0] // weights.shape[0], weights.shape[0])))
        for i, col in enumerate(table.column_names):
            if pat.is_float64(table.column(i).type):
                typed_w = w.astype(np.float64)
            else:
                typed_w = w.astype(np.float32)

            tmp = pac.multiply_checked(table.column(i), typed_w)
            table = table.set_column(i, col, tmp)

        for col in pop_weighted_cols:
            if pat.is_float64(pop_weight_table[col].type):
                typed_w = table["total_population"].to_numpy().astype(np.float64)
            else:
                typed_w = table["total_population"].to_numpy().astype(np.float32)
            tmp = pac.multiply_checked(pop_weight_table[col], typed_w)
            table = table.append_column(col, tmp)

        for level in args.levels:
            all_q_data = {}
            for col in table.column_names:  # TODO can we do all at once since we dropped date?
                all_q_data[col] = pa_array_quantiles(table[col], level)

            # all_q_data = {col: pa_array_quantiles(table[col]) for col in table.column_names}

            # we could do this outside the date loop and cache for each adm level...
            out_shape = (len(percentiles),) + adm_level_values[level].shape
            all_q_data[level] = np.broadcast_to(adm_level_values[level], out_shape)
            all_q_data["date"] = np.broadcast_to(date, out_shape)
            all_q_data["quantile"] = np.broadcast_to(quantiles[..., None], out_shape)

            for col in per_capita_cols:
                all_q_data[col + "_per_100k"] = 100000.0 * all_q_data[col] / all_q_data["total_population"]

            for col in pop_weighted_cols:
                all_q_data[col] = all_q_data[col] / all_q_data["total_population"]

            for col in all_q_data:
                all_q_data[col] = xp.to_cpu(all_q_data[col].T.ravel())

            write_queue.put((os.path.join(output_dir, level + "_quantiles.csv"), all_q_data))

        del dataset
        gc.collect()

    write_queue.put(None)  # send signal to term loop
    write_thread.join()  # join the write_thread


if __name__ == "__main__":
    # from line_profiler import LineProfiler

    # lp = LineProfiler()
    # lp_wrapper = lp(main)
    # lp.add_function(main._process_date)
    # lp_wrapper()
    # lp.print_stats()
    main()
