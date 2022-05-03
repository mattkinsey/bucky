"""Postprocesses data across dates and simulation runs before aggregating at geographic levels (ADM0, ADM1, or ADM2)."""
import gc
import queue
import shutil
import threading

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pac
import pyarrow.dataset as ds
import pyarrow.types as pat
import tqdm
from loguru import logger

from .numerical_libs import enable_cupy, reimport_numerical_libs, xp
from .util.util import _banner

# TODO switch to cupy.quantile instead of percentile (they didn't have that when we first wrote this)
# also double check but the api might be consistant by now so we dont have to handle numpy/cupy differently


def main(cfg):
    """Main method for postprocessing the raw outputs from an MC run."""

    _banner("Postprocessing Quantiles")

    quantiles = cfg["postprocessing.output_quantiles"]
    verbose = cfg["runtime.verbose"]
    use_gpu = cfg["runtime.use_cupy"]
    run_dir = cfg["postprocessing.run_dir"]
    data_dir = run_dir / "data"
    metadata_dir = run_dir / "metadata"
    output_levels = cfg["postprocessing.output_levels"]

    # if verbose:
    #    logger.info(cfg)

    output_dir = cfg["postprocessing.output_dir"]
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Copy metadata
    output_metadata_dir = output_dir / "metadata"
    output_metadata_dir.mkdir(exist_ok=True)
    # TODO this should probably recurse directories too...
    for md_file in metadata_dir.iterdir():
        shutil.copy2(md_file, output_metadata_dir / md_file.name)

    adm_mapping = pd.read_csv(metadata_dir / "adm_mapping.csv")
    dates = pd.read_csv(metadata_dir / "dates.csv")
    dates = dates["date"].to_numpy()

    n_adm2 = len(adm_mapping)
    adm2_sorted_ind = xp.argsort(xp.array(adm_mapping["adm2"].to_numpy()))

    if cfg["runtime.use_cupy"]:
        enable_cupy(optimize=True)
        reimport_numerical_libs("postprocess")

    # TODO move these to cfg?
    per_capita_cols = set(cfg["postprocessing.per_capita_cols"])
    pop_weighted_cols = set(cfg["postprocessing.pop_weighted_cols"])

    adm_mapping["adm0"] = 1
    adm_map = adm_mapping.to_dict(orient="list")
    adm_map = {k: xp.array(v)[adm2_sorted_ind] for k, v in adm_map.items()}
    adm_array_map = {k: xp.unique(v, return_inverse=True)[1] for k, v in adm_map.items()}
    adm_sizes = {k: xp.to_cpu(xp.max(v) + 1).item() for k, v in adm_array_map.items()}
    adm_level_values = {k: xp.to_cpu(xp.unique(v)) for k, v in adm_map.items()}
    adm_level_values["adm0"] = np.array(["US"])

    # TODO fix lookup tables
    # f args.lookup is not None and "weight" in lookup_df.columns:
    #   weight_series = lookup_df.set_index("adm2")["weight"].reindex(adm_mapping["adm2"], fill_value=0.0)
    #   weights = np.array(weight_series.to_numpy(), dtype=np.float32)
    #   # TODO we should ignore all the adm2 not in weights rather than just 0ing them (it'll go alot faster)
    # lse:
    weights = np.ones_like(adm2_sorted_ind, dtype=np.float32)

    # TODO switch to using the async_thread/buckyoutputwriter util
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

        q_data_gpu = xp.empty((len(percentiles), adm_sizes[level]), dtype=data_gpu.dtype)

        if adm_sizes[level] == 1:
            # TODO need switching here b/c cupy handles xp.percentile weird with a size 1 dim :(
            if use_gpu:
                level_data_gpu = xp.sum(data_gpu, axis=0)  # need this if cupy
            else:
                level_data_gpu = xp.sum(data_gpu, axis=0, keepdims=True).T  # for numpy
            # q_data_gpu = xp.empty((len(percentiles), adm_sizes[level]), dtype=level_data_gpu.dtype)
            # It appears theres a cupy bug when the 1st axis of the array passed to percentiles has size 1
            xp.percentile(level_data_gpu, q=percentiles, axis=0, out=q_data_gpu)
        else:
            level_data_gpu = xp.zeros((adm_sizes[level], data_gpu.shape[1]), dtype=data_gpu.dtype)
            xp.scatter_add(level_data_gpu, adm_array_map[level], data_gpu)
            # q_data_gpu = xp.empty((len(percentiles), adm_sizes[level]), dtype=level_data_gpu.dtype)
            xp.percentile(level_data_gpu, q=percentiles, axis=1, out=q_data_gpu)
        return q_data_gpu

    try:
        percentiles = xp.array(quantiles, dtype=np.float64) * 100.0
        quantiles = np.array(quantiles)

        def get_nrids():
            dataset = ds.dataset(data_dir, format="parquet", partitioning=["date"])
            rids = dataset.to_table(columns=["rid"])
            return len(pac.unique(rids.column(0)))

        logger.opt(lazy=True).info("Found {n_rids} unique monte carlos", n_rids=lambda: get_nrids())
        for date_i, date in enumerate(tqdm.tqdm(dates)):
            dataset = ds.dataset(data_dir, format="parquet", partitioning=["date"])
            table = dataset.to_table(filter=ds.field("date") == "date=" + str(date_i))
            table = table.drop(("date", "rid", "adm2_id"))  # we don't need these b/c metadata

            pop_weighted_cols = pop_weighted_cols.intersection(table.column_names)
            pop_weight_table = table.select(pop_weighted_cols)
            table = table.drop(pop_weighted_cols)

            w = np.ravel(np.broadcast_to(weights, (table.shape[0] // weights.shape[0], weights.shape[0])))
            for i, col in enumerate(table.column_names):
                if pat.is_float64(table.column(i).type):
                    typed_w = w.astype(np.float64)
                else:
                    typed_w = w.astype(np.float32)

                tmp = pac.multiply_checked(table.column(i), typed_w)  # pylint: disable=no-member
                table = table.set_column(i, col, tmp)

            for col in pop_weighted_cols:

                if pat.is_float64(pop_weight_table[col].type):
                    typed_w = table["total_population"].to_numpy().astype(np.float64)
                else:
                    typed_w = table["total_population"].to_numpy().astype(np.float32)
                tmp = pac.multiply_checked(pop_weight_table[col], typed_w)  # pylint: disable=no-member
                table = table.append_column(col, tmp)

            for level in output_levels:
                all_q_data = {}
                for col in table.column_names:  # TODO can we do all at once since we dropped date?
                    all_q_data[col] = pa_array_quantiles(table[col], level)

                # we could do this outside the date loop and cache for each adm level...
                out_shape = (len(percentiles),) + adm_level_values[level].shape
                all_q_data[level] = np.broadcast_to(adm_level_values[level], out_shape)
                all_q_data["date"] = np.broadcast_to(date, out_shape)
                all_q_data["quantile"] = np.broadcast_to(quantiles[..., None], out_shape)

                for col in per_capita_cols:
                    if col in all_q_data:
                        all_q_data[col + "_per_100k"] = 100000.0 * all_q_data[col] / all_q_data["total_population"]

                for col in pop_weighted_cols:
                    all_q_data[col] = all_q_data[col] / all_q_data["total_population"]

                for col in all_q_data:
                    all_q_data[col] = xp.to_cpu(all_q_data[col].T.ravel())

                write_queue.put((output_dir / (level + "_quantiles.csv"), all_q_data))

            del dataset
            gc.collect()

    except (KeyboardInterrupt, SystemExit):
        logger.warning("Caught SIGINT, cleaning up")
        write_queue.put(None)  # send signal to term loop
        write_thread.join()  # join the write_thread
    finally:
        write_queue.put(None)  # send signal to term loop
        write_thread.join()  # join the write_thread
