"""TODO: Summary
"""
# TODO break into file now that we have the utils submodule (also update __init__)

import copy
import logging

# import lz4.frame
import lzma  # lzma is slow but reallllly gets that file size down...
import os
import pickle
import threading

import numpy as np
import pandas as pd
import tqdm


# https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
class TqdmLoggingHandler(logging.Handler):

    """TODO: Summary"""

    def __init__(self, level=logging.NOTSET):  # pylint: disable=useless-super-delegation
        """TODO: Summary

        Parameters
        ----------
        level : TYPE, optional
            TODO: Description
        """
        super().__init__(level)

    def emit(self, record):
        """TODO: Summary

        Parameters
        ----------
        record : TYPE
            TODO: Description
        """
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):  # pylint: disable=try-except-raise
            raise
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)


class dotdict(dict):
    """dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        """TODO: Summary

        Parameters
        ----------
        memo : None, optional
            TODO: Description

        Returns
        -------
        TYPE
            TODO: Description
        """
        return dotdict({key: copy.deepcopy(value) for key, value in self.items()})


def remove_chars(seq):
    """TODO: Summary

    Parameters
    ----------
    seq : TYPE
        TODO: Description

    Returns
    -------
    TYPE
        TODO: Description
    """
    seq_type = type(seq)
    if seq_type != str:
        return seq

    return seq_type().join(filter(seq_type.isdigit, seq))


def map_np_array(a, d):
    """TODO: Summary

    Parameters
    ----------
    a : TYPE
        TODO: Description
    d : TYPE
        TODO: Description

    Returns
    -------
    TYPE
        TODO: Description
    """
    n = np.ndarray(a.shape)
    for k in d:
        n[a == k] = d[k]
    return n


def estimate_IFR(age):
    """TODO: Summary

    Parameters
    ----------
    age : TYPE
        TODO: Description

    Returns
    -------
    TYPE
        TODO: Description
    """
    # from best fit in https://www.medrxiv.org/content/10.1101/2020.07.23.20160895v4.full.pdf
    # std err is 0.17 on the const and 0.003 on the linear term
    return np.exp(-7.56 + 0.121 * age) / 100.0


def bin_age_csv(filename, out_filename):
    """TODO: Summary

    Parameters
    ----------
    filename : TYPE
        TODO: Description
    out_filename : TYPE
        TODO: Description
    """
    df = pd.read_csv(filename, header=None, names=["fips", "age", "N"])
    pop_weighted_IFR = df.N.to_numpy() * estimate_IFR(df.age.to_numpy())
    df = df.assign(IFR=pop_weighted_IFR)
    df["age_group"] = pd.cut(df["age"], np.append(np.arange(0, 76, 5), 120), right=False)
    df = df.groupby(["fips", "age_group"]).sum()[["N", "IFR"]].unstack("age_group")

    df = df.assign(IFR=df.IFR / df.N)

    df.to_csv(out_filename)


def date_to_t_int(dates, start_date):
    """TODO: Summary

    Parameters
    ----------
    dates : TYPE
        TODO: Description
    start_date : TYPE
        TODO: Description

    Returns
    -------
    TYPE
        TODO: Description
    """
    return np.array([(date - start_date).days for date in dates], dtype=int)


def _cache_files(fname_list, cache_name):
    """TODO: Summary

    Parameters
    ----------
    fname_list : TYPE
        TODO: Description
    cache_name : TYPE
        TODO: Description
    """
    tmp = {f: open(f, "rb").read() for f in fname_list}
    with lzma.open("run_cache/" + cache_name + ".p.xz", "wb") as f:
        pickle.dump(tmp, f)


def cache_files(*argv):
    """TODO: Summary

    Parameters
    ----------
    *argv
        TODO: Description
    """
    thread = threading.Thread(target=_cache_files, args=argv)
    thread.start()


def unpack_cache(cache_file):
    """TODO: Summary

    Parameters
    ----------
    cache_file : TYPE
        TODO: Description
    """
    with lzma.open(cache_file, "rb") as f:
        tmp = pickle.load(f)  # nosec

    # os.mkdir(cache_file[:-5])
    for fname in tmp:
        new_file = cache_file[:-5] + "/" + fname
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        with open(new_file, "wb") as f:
            f.write(tmp[fname])


def _banner():
    """TODO: Summary"""
    print(r" ____             _          ")  # noqa: T001
    print(r"| __ ) _   _  ___| | ___   _ ")  # noqa: T001
    print(r"|  _ \| | | |/ __| |/ / | | |")  # noqa: T001
    print(r"| |_) | |_| | (__|   <| |_| |")  # noqa: T001
    print(r"|____/ \__,_|\___|_|\_\\__, |")  # noqa: T001
    print(r"                       |___/ ")  # noqa: T001
    print(r"                             ")  # noqa: T001
