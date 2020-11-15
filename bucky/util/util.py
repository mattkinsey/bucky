# TODO break into file now that we have the utils submodule (also update __init__)

import copy
import logging

import numpy as np
import pandas as pd
import tqdm


# https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):  # pylint: disable=useless-super-delegation
        super().__init__(level)

    def emit(self, record):
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
        """return a deepcopy of the dict"""
        return dotdict({key: copy.deepcopy(value) for key, value in self.items()})


def remove_chars(seq):
    seq_type = type(seq)
    if seq_type != str:
        return seq

    return seq_type().join(filter(seq_type.isdigit, seq))


def map_np_array(a, d):
    n = np.ndarray(a.shape)
    for k in d:
        n[a == k] = d[k]
    return n


def estimate_IFR(age):
    # from best fit in https://www.medrxiv.org/content/10.1101/2020.07.23.20160895v4.full.pdf
    # std err is 0.17 on the const and 0.003 on the linear term
    return np.exp(-7.56 + 0.121 * age) / 100.0


def bin_age_csv(filename, out_filename):
    df = pd.read_csv(filename, header=None, names=["fips", "age", "N"])
    pop_weighted_IFR = df.N.to_numpy() * estimate_IFR(df.age.to_numpy())
    df = df.assign(IFR=pop_weighted_IFR)
    df["age_group"] = pd.cut(df["age"], np.append(np.arange(0, 76, 5), 120), right=False)
    df = df.groupby(["fips", "age_group"]).sum()[["N", "IFR"]].unstack("age_group")

    df = df.assign(IFR=df.IFR / df.N)

    df.to_csv(out_filename)


def date_to_t_int(dates, start_date):
    return np.array([(date - start_date).days for date in dates], dtype=int)


def _banner():
    """It's a banner for the CLI..."""
    print(r" ____             _          ")  # noqa: T001
    print(r"| __ ) _   _  ___| | ___   _ ")  # noqa: T001
    print(r"|  _ \| | | |/ __| |/ / | | |")  # noqa: T001
    print(r"| |_) | |_| | (__|   <| |_| |")  # noqa: T001
    print(r"|____/ \__,_|\___|_|\_\\__, |")  # noqa: T001
    print(r"                       |___/ ")  # noqa: T001
    print(r"                             ")  # noqa: T001
