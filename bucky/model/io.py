"""Monte carlo output handler"""
import datetime
import queue
import threading
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pap

from ..numerical_libs import sync_numerical_libs, xp
from ..util.async_thread import AsyncQueueThread


@sync_numerical_libs
def init_write_thread(**kwargs):
    """Init write thread w/ a nonblocking stream"""
    stream = xp.cuda.Stream(non_blocking=True) if xp.is_cupy else None
    pinned_mem = {}
    return {"stream": stream, "pinned_mem": pinned_mem}


@sync_numerical_libs
def write_parquet_dataset(df_data, data_dir, stream, pinned_mem):
    """Write a dataframe of MC output to parquet"""
    for k, v in df_data.items():
        if k not in pinned_mem:
            pinned_mem[k] = xp.empty_like_pinned(v)

        xp.to_cpu(v, stream=stream, out=pinned_mem[k])

    if stream is not None:
        stream.synchronize()

    pa_data = {k: pa.array(v) for k, v in pinned_mem.items()}
    table = pa.table(pa_data)
    pap.write_to_dataset(table, data_dir, partition_cols=["date"])


'''
@sync_numerical_libs
def parquet_writer(write_queue):
    """Write thread loop that pulls from an async queue and writes to parquet"""
    # Call to_write.get() until it returns None
    stream = xp.cuda.Stream(non_blocking=True) if xp.is_cupy else None
    pinned_mem = {}
    for base_fname, df_data in iter(write_queue.get, None):
        for k, v in df_data.items():
            if k not in pinned_mem:
                pinned_mem[k] = xp.empty_like_pinned(v)

            xp.to_cpu(v, stream=stream, out=pinned_mem[k])

        if stream is not None:
            stream.synchronize()

        pa_data = {k: pa.array(v) for k, v in pinned_mem.items()}
        table = pa.table(pa_data)
        pap.write_to_dataset(table, base_fname, partition_cols=["date"])
'''


class BuckyOutputWriter:
    def __init__(self, output_base_dir, run_id, data_format="parquet"):
        """Init the writer globals"""
        self.output_dir = Path(output_base_dir) / str(run_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self.output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        if data_format == "parquet":
            self.write_thread = AsyncQueueThread(write_parquet_dataset, pre_func=init_write_thread, data_dir=data_dir)

    @sync_numerical_libs
    def write_metadata(self, g_data, t_max):
        """Write metadata to output dir"""
        metadata_dir = self.output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        # write out adm level mappings
        adm_map_file = metadata_dir / "adm_mapping.csv"
        adm2_ids = xp.to_cpu(g_data.adm2_id)
        adm1_ids = xp.to_cpu(g_data.adm1_id)
        adm0_ids = np.broadcast_to(g_data.adm0_name, adm2_ids.shape)
        adm_map_table = np.stack([adm2_ids, adm1_ids, adm0_ids]).T
        np.savetxt(adm_map_file, adm_map_table, header="adm2,adm1,adm0", comments="", delimiter=",", fmt="%s")

        # write out dates
        date_file = metadata_dir / "dates.csv"
        str_dates = [str(g_data.start_date + datetime.timedelta(days=int(np.round(t)))) for t in range(t_max + 1)]
        np.savetxt(date_file, str_dates, header="date", comments="", delimiter=",", fmt="%s")

    def write_mc_data(self, data_dict):
        """Write the data from one MC to the output dir"""
        # flatten the shape
        for c in data_dict:
            data_dict[c] = data_dict[c].ravel()

        # push the data off to the write thread
        self.write_thread.put(data_dict)

    def write_params(self, seed, params):
        """TODO WIP Write MC parameters per iteration"""
        # TODO
        metadata_dir = self.output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        params_base_dir = metadata_dir / "params"

        # could also flatten the dict and savez it?

        # TODO rewrite as a recursive func to handle deeper nesting
        for k, v in params.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    param_dir = params_base_dir / k / k2
                    param_dir.mkdir(parents=True, exist_ok=True)
                    f = param_dir / (str(seed) + ".npy")
                    # print(f)
                    xp.save(f, v2)
            else:
                param_dir = params_base_dir / k
                param_dir.mkdir(parents=True, exist_ok=True)
                f = param_dir / (str(seed) + ".npy")
                # print(f)
                xp.save(f, v)

        # from IPython import embed
        # embed()
        pass

    def write_historical_data(self):
        """TODO Write historical data used"""
        # TODO
        pass

    def write_par_files(self):
        """TODO Copy parameter specs"""
        # TODO
        pass

    def close(self):
        """Cleanup and join write thread"""
        self.write_thread.close()
