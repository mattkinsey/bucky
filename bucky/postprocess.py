"""Postprocesses data across dates and simulation runs before aggregating at geographic levels (ADM0, ADM1, or ADM2)."""
import concurrent.futures
import gc
import queue
import shutil
import threading

import numpy as np
import pandas as pd
import tqdm
from fastparquet import ParquetFile
from loguru import logger

from .numerical_libs import enable_cupy, reimport_numerical_libs, xp
from .util.util import _banner

# TODO switch to cupy.quantile instead of percentile (they didn't have that when we first wrote this)
# also double check but the api might be consistant by now so we dont have to handle numpy/cupy differently


def main(cfg):
    """Main method for postprocessing the raw outputs from an MC run."""

    _banner("Postprocessing Quantiles")

    # verbose = cfg["runtime.verbose"]
    # use_gpu = cfg["runtime.use_cupy"]
    run_dir = cfg["postprocessing.run_dir"]
    data_dir = run_dir / "data"
    metadata_dir = run_dir / "metadata"

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

    if cfg["runtime.use_cupy"]:
        enable_cupy(optimize=True)
        reimport_numerical_libs("postprocess")

    # TODO switch to using the async_thread/buckyoutputwriter util
    write_queue = queue.Queue()

    def _writer():
        """Write thread that will pull from a queue."""
        # Call to_write.get() until it returns None
        file_tables = {}
        for fname, q_dict in iter(write_queue.get, None):
            df = pd.DataFrame(q_dict)
            id_col = df.columns[df.columns.str.contains("adm.")].values[0]
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index([id_col, "date", "quantile"])
            df = df.reindex(sorted(df.columns), axis=1)
            if fname in file_tables:
                file_tables[fname].append(df)
            else:
                file_tables[fname] = [
                    df,
                ]
            write_queue.task_done()

        # dump tables to disk
        for fname in tqdm.tqdm(file_tables):
            df = pd.concat(file_tables[fname])
            df = df.sort_index()
            df.to_csv(fname, header=True, mode="w", date_format="%Y-%m-%d")
        write_queue.task_done()

    write_thread = threading.Thread(target=_writer)
    write_thread.start()

    adm2_sorted_ind = np.argsort(adm_mapping["adm2"].to_numpy())
    adm_map = adm_mapping.to_dict(orient="list")
    adm_map = {k: np.array(v)[adm2_sorted_ind] for k, v in adm_map.items()}
    adm_array_map = {k: np.unique(v, return_inverse=True)[1] for k, v in adm_map.items()}
    adm_sizes = {k: np.max(v) + 1 for k, v in adm_array_map.items()}
    adm_level_values = {k: np.unique(v) for k, v in adm_map.items()}
    adm_level_values["adm0"] = np.array(["US"])

    # pf = ParquetFile(str(data_dir))

    def process_one_date(date_enum_tuple):
        nonlocal cfg, write_queue, adm_sizes, adm_array_map, adm2_sorted_ind, adm_level_values  # pf
        date_i = date_enum_tuple[0]
        date = date_enum_tuple[1]

        pf = ParquetFile(str(data_dir))
        date_filter = [("date", "==", date_i)]
        data_cols = sorted(set(pf.columns).difference(("date", "rid", "adm2_id")))

        pop_weighted_cols = set(cfg["postprocessing.pop_weighted_cols"])
        per_capita_cols = set(cfg["postprocessing.per_capita_cols"])

        pop_weighted_cols = pop_weighted_cols.intersection(pf.columns)
        per_capita_cols = per_capita_cols.intersection(pf.columns)

        with xp.cuda.Device(date_i % cfg["postprocessing.n_gpu"]), xp.cuda.Stream(non_blocking=True):

            quantiles = xp.array(cfg["postprocessing.output_quantiles"])
            total_population = xp.array(pf.to_pandas(["total_population"], filters=date_filter).to_numpy()).T[0]
            pd_data = pf.to_pandas(data_cols, filters=date_filter)
            pd_columns = pd_data.columns

            all_data = xp.array(pd_data.to_numpy()).T
            for c_ind, col in enumerate(pd_columns):
                if col in pop_weighted_cols:
                    all_data[c_ind] *= total_population

            all_data = all_data.reshape(len(data_cols), -1, adm_sizes["adm2"])
            all_data = all_data[..., adm2_sorted_ind]
            all_data = xp.moveaxis(all_data, source=2, destination=0).copy()

            for level in cfg["postprocessing.output_levels"]:
                level_data = xp.zeros((adm_sizes[level],) + all_data.shape[1:], dtype=all_data.dtype)
                xp.scatter_add(level_data, adm_array_map[level], all_data)
                data_quantiles = xp.quantile(level_data, q=quantiles, axis=-1)

                all_q_data = {}
                for c_ind, col in enumerate(data_cols):
                    all_q_data[col] = data_quantiles[..., c_ind]

                for col in per_capita_cols:
                    if col in per_capita_cols:
                        all_q_data[col + "_per_100k"] = 100000.0 * all_q_data[col] / all_q_data["total_population"]

                for col in pop_weighted_cols:
                    all_q_data[col] = all_q_data[col] / all_q_data["total_population"]

                out_shape = (len(quantiles),) + adm_level_values[level].shape
                all_q_data[level] = np.broadcast_to(adm_level_values[level], out_shape)
                all_q_data["date"] = np.broadcast_to(date, out_shape)
                all_q_data["quantile"] = np.broadcast_to(quantiles[..., None], out_shape)

                for col in all_q_data:
                    all_q_data[col] = xp.to_cpu(all_q_data[col].T.ravel())

                write_queue.put((output_dir / (level + "_quantiles.csv"), all_q_data))

    # TODO if n_cpu == 1:
    # for date_tuple in enumerate(tqdm.tqdm(dates)):
    #    process_one_date(date_tuple)

    with tqdm.tqdm(total=len(dates)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["postprocessing.n_cpu"]) as executor:
            try:
                futures = [executor.submit(process_one_date, date_tuple) for date_tuple in enumerate(dates)]
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)

            except (KeyboardInterrupt, SystemExit):
                logger.warning("Caught SIGINT, cleaning up")
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                except (KeyboardInterrupt, SystemExit):
                    logger.warning("Caught second SIGINT, force closing threads")
                    executor.shutdown(wait=False, cancel_futures=True)
                pbar.close()
                write_queue.put(None)  # send signal to term loop
                write_thread.join()  # join the write_thread
            finally:
                write_queue.put(None)  # send signal to term loop
                write_thread.join()  # join the write_thread

    write_queue.put(None)
    write_thread.join()
