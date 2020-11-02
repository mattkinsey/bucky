"""Provides an interface to import numerical libraries using the GPU (if available)."""

import contextlib

# Default imports for cpu code
# This will be overwritten with a call to .numerical_libs.use_cupy()
import numpy as xp
import scipy.integrate._ivp.ivp as ivp  # noqa: F401
import scipy.sparse as sparse  # noqa: F401

xp.scatter_add = xp.add.at
xp.optimize_kernels = contextlib.nullcontext
xp.to_cpu = lambda x, **kwargs: x  # one arg noop


class ExperimentalWarning(Warning):
    pass


xp.ExperimentalWarning = ExperimentalWarning


def use_cupy(optimize=False):
    """Perform imports for libraries with APIs matching numpy, scipy.integrate.ivp, scipy.sparse.

    These imports will use a monkey-patched version of these modules
    that has had all it's numpy references replaced with CuPy.

    if optimize is True, place the kernel optimization context in xp.optimize_kernels,
    otherwise make it a nullcontext (noop)

    returns nothing but imports a version of 'xp', 'ivp', and 'sparse' to the global scope of this module

    Parameters
    ----------
    optimize : bool
        Enable kernel optimization in cupy >=v8.0.0. This will slow down initial
        function call (mostly reduction operations) but will offer better
        performance for repeated calls (e.g. in the RHS call of an integrator).

    Returns
    -------
    exit_code : int
        Non-zero value indicates error code, or zero on success.

    Raises
    ------
    NotImplementedError
        If the user calls a monkeypatched function of the libs that isn't
        fully implemented.

    """
    import importlib
    import logging

    cupy_spec = importlib.util.find_spec("cupy")
    if cupy_spec is None:
        logging.info("CuPy not found, reverting to cpu/numpy")
        return 1

    global xp, ivp, sparse

    if xp.__name__ == "cupy":
        logging.info("CuPy already loaded, skipping")
        return 0

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

    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.memory.malloc_managed).malloc)

    # add cupy search sorted for scipy.ivp (this was only added to cupy sometime between v6.0.0 and v7.0.0)
    if ~hasattr(cp, "searchsorted"):
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

    spec = importlib.util.find_spec("optuna")
    if spec is None:
        logging.info("Optuna not installed, kernel opt is disabled")
        cp.optimize_kernels = contextlib.nullcontext
        cp.ExperimentalWarning = ExperimentalWarning
    elif optimize:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARN)
        logging.info("Using optuna to optimize kernels, the first calls will be slowwwww")
        cp.optimize_kernels = cupyx.optimizing.optimize
        cp.ExperimentalWarning = optuna.exceptions.ExperimentalWarning
    else:
        cp.optimize_kernels = contextlib.nullcontext
        cp.ExperimentalWarning = ExperimentalWarning

    def cp_to_cpu(x, stream=None, out=None):
        if "cupy" in type(x).__module__:
            return x.get(stream=stream, out=out)
        else:
            return x

    cp.to_cpu = cp_to_cpu  # lambda x, **kwargs: x.get(**kwargs) if "cupy" in type(x).__module__ else x

    xp = cp
    import cupyx.scipy.sparse as sparse

    # TODO need to check cupy version is >9.0.0a1 in order to use sparse

    return 0
