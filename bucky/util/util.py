# TODO break into file now that we have the utils submodule (also update __init__)

import copy

# import lz4.frame
import lzma  # lzma is slow but reallllly gets that file size down...
import os
import pickle
import threading

import numpy as np
import pandas as pd


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return dotdict({key: copy.deepcopy(value) for key, value in self.items()})


def map_np_array(a, d):
    n = np.ndarray(a.shape)
    for k in d:
        n[a == k] = d[k]
    return n

def estimate_IFR(age):
    # from best fit in https://www.medrxiv.org/content/10.1101/2020.07.23.20160895v4.full.pdf
    #std err is 0.17 on the const and 0.003 on the linear term
    return np.exp(-7.56 + 0.121 * age)/100.

def bin_age_csv(filename, out_filename):
    df = pd.read_csv(filename, header=None, names=["fips", "age", "N"])
    pop_weighted_IFR = df.N.to_numpy()*estimate_IFR(df.age.to_numpy())
    df = df.assign(IFR=pop_weighted_IFR)
    df["age_group"] = pd.cut(
        df["age"], np.append(np.arange(0, 76, 5), 120), right=False
    )
    df = df.groupby(["fips", "age_group"]).sum()[["N", "IFR"]].unstack("age_group")

    df = df.assign(IFR=df.IFR/df.N)

    df.to_csv(out_filename)

def date_to_t_int(dates, start_date):
        return np.array([(date - start_date).days for date in dates], dtype=int)

def _cache_files(fname_list, cache_name):
    tmp = {f: open(f, "rb").read() for f in fname_list}
    with lzma.open("run_cache/" + cache_name + ".p.xz", "wb") as f:
        pickle.dump(tmp, f)


def cache_files(*argv):
    thread = threading.Thread(target=_cache_files, args=argv)
    thread.start()


def unpack_cache(cache_file):
    with lzma.open(cache_file, "rb") as f:
        tmp = pickle.load(f)

    # os.mkdir(cache_file[:-5])
    for fname in tmp:
        new_file = cache_file[:-5] + "/" + fname
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        with open(new_file, "wb") as f:
            f.write(tmp[fname])


def import_numerical_libs(gpu=False):
    """ Perform imports for libraries with APIs matching numpy, scipy.integrate.ivp, scipy.sparse

    if gpu is True, this these imports will use a monkey-patched version of these modules that has had all it's numpy references replaced with CuPy

    if gpu is False, simply performs:
        import scipy.integrate._ivp.ivp as ivp
        import numpy as xp
        import scipy.sparse as sparse

    returns nothing but imports a version of 'xp', 'ivp', and 'sparse' at the global scope
    """
    global xp, ivp, sparse
    if gpu:
        import importlib
        import sys

        # modify src before importing
        def modify_and_import(module_name, package, modification_func):
            spec = importlib.util.find_spec(module_name, package)
            source = spec.loader.get_source(module_name)
            new_source = modification_func(source)
            module = importlib.util.module_from_spec(spec)
            codeobj = compile(new_source, module.__spec__.origin, "exec")
            exec(codeobj, module.__dict__)
            sys.modules[module_name] = module
            return module

        import cupy as cp

        # add cupy search sorted for scipy.ivp (this was only added to cupy sometime between v6.0.0 and v7.0.0)
        if ~hasattr(cp, 'searchsorted'):
            # NB: this isn't correct in general but it works for what scipy solve_ivp needs...
            def cp_searchsorted(a, v, side="right", sorter=None):
                if side != "right":
                    raise NotImplementedError
                if sorter is not None:
                    raise NotImplementedError  # sorter = list(range(len(a)))
                tmp = v >= a
                if cp.all(tmp):
                    return len(a)
                return cp.argmax(~tmp)

            cp.searchsorted = cp_searchsorted

        for name in ("common", "base", "rk", "ivp"):
            ivp = modify_and_import(
                "scipy.integrate._ivp." + name,
                None,
                lambda src: src.replace("import numpy", "import cupy"),
            )

        import cupyx
        cp.scatter_add = cupyx.scatter_add

        xp = cp
        import cupyx.scipy.sparse as sparse
    else:
        import scipy.integrate._ivp.ivp as ivp
        import numpy as xp
        import scipy.sparse as sparse
        xp.scatter_add = xp.add.at


# TODO we should monkeypatch this
def truncnorm(xp, loc=0.0, scale=1.0, size=1, a_min=None, a_max=None):
    """ Provides a truncnorm implementation that is compatible with cupy
    """
    ret = xp.random.normal(loc, scale, size)
    if a_min is None:
        a_min = -xp.inf
    if a_max is None:
        a_max = xp.inf

    while True:
        valid = (ret > a_min) & (ret < a_max)
        if valid.all():
            return ret
        ret[~valid] = xp.random.normal(loc, scale, ret[~valid].shape)

def force_cpu(var):
    return var.get() if "cupy" in type(var).__module__ else var

def _banner():
    print(r" ____             _          ")
    print(r"| __ ) _   _  ___| | ___   _ ")
    print(r"|  _ \| | | |/ __| |/ / | | |")
    print(r"| |_) | |_| | (__|   <| |_| |")
    print(r"|____/ \__,_|\___|_|\_\\__, |")
    print(r"                       |___/ ")
    print(r"                             ")
