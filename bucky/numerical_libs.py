"""Provides an interface to import numerical libraries using the GPU (if available).

The main goal of this is to smooth over the differences between numpy and cupy so that
the rest of the code can use them interchangably. We also need to  monkey patch scipy's ivp solver
to work on cupy arrays.

Notes
-----
Linters **HATE** this module because it's really abusing the import system (by design).

"""

import contextlib
import importlib
import inspect
from functools import wraps

import numpy as xp
import scipy.integrate._ivp.ivp as xp_ivp
import scipy.sparse as xp_sparse
import scipy.special

# Default imports for cpu code
# This will be overwritten with a call to .numerical_libs.enable_cupy()
import bucky

xp.scatter_add = xp.add.at
xp.optimize_kernels = contextlib.nullcontext
xp.to_cpu = lambda x, **kwargs: x  # noop
xp.special = scipy.special
xp.empty_pinned = xp.empty
xp.empty_like_pinned = xp.empty_like
xp.zeros_pinned = xp.zeros
xp.zeros_like_pinned = xp.zeros_like


bucky.xp = xp
bucky.xp_sparse = xp_sparse
bucky.xp_ivp = xp_ivp


class MockExperimentalWarning(Warning):
    """Simple class to mock the optuna warning if we don't have optuna."""


xp.ExperimentalWarning = MockExperimentalWarning


reimport_cache = set()


def reimport_numerical_libs(context=None):
    """Reimport xp, xp_sparse, xp_ivp from the global context (in case they've been update to cupy)."""
    global reimport_cache  # pylint: disable=global-statement
    if context in reimport_cache:
        return
    caller_globals = dict(inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
    for lib in ("xp", "xp_sparse", "xp_ivp"):
        if lib in caller_globals:
            bucky_module = importlib.import_module("bucky")
            caller_globals[lib] = getattr(bucky_module, lib)
    if context is not None:
        reimport_cache.add(context)


def sync_numerical_libs(func):
    """Decorator that pulls xp, xp_sparse, xp_ivp from the global bucky context into the wrapped function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper that checks if we've already overridden this functions imports."""
        global reimport_cache  # pylint: disable=global-statement
        context = func.__qualname__
        if context in reimport_cache:
            return func(*args, **kwargs)
        for lib in ("xp", "xp_sparse", "xp_ivp"):
            if lib in func.__globals__:
                bucky_module = importlib.import_module("bucky")
                func.__globals__[lib] = getattr(bucky_module, lib)

        reimport_cache.add(context)
        return func(*args, **kwargs)

    return wrapper


def enable_cupy(optimize=False):
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
    import logging  # pylint: disable=import-outside-toplevel
    import sys  # pylint: disable=import-outside-toplevel

    cupy_spec = importlib.util.find_spec("cupy")
    if cupy_spec is None:
        logging.info("CuPy not found, reverting to cpu/numpy")
        return 1

    if xp.__name__ == "cupy":
        logging.info("CuPy already loaded, skipping")
        return 0

    # modify src before importing
    def modify_and_import(module_name, package, modification_func):
        """Return an imported class after applying the modification function to the source files."""
        spec = importlib.util.find_spec(module_name, package)
        source = spec.loader.get_source(module_name)
        new_source = modification_func(source)
        module = importlib.util.module_from_spec(spec)
        codeobj = compile(new_source, module.__spec__.origin, "exec")
        exec(codeobj, module.__dict__)  # pylint: disable=exec-used
        sys.modules[module_name] = module
        return module

    import cupy as cp  # pylint: disable=import-outside-toplevel

    # import numpy as np  # pylint: disable=import-outside-toplevel, reimported

    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.memory.malloc_managed).malloc)
    # cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)

    def scipy_import_replacement(src):
        """Perform the required numpy->cupy str replacements on the scipy source files."""
        # replace numpy w/ cupy
        src = src.replace("import numpy", "import cupy")
        # fix a call to searchsorted by making sure it's params are typed correctly for the cupy version
        src = src.replace(
            "t_eval_i_new = np.searchsorted(t_eval, t, side='right')",
            "t_eval_i_new = np.searchsorted(t_eval, np.array([t]), side='right')",
        )

        return src

    for name in ("common", "base", "rk", "ivp"):
        bucky.xp_ivp = modify_and_import(
            "scipy.integrate._ivp." + name,
            None,
            scipy_import_replacement,
        )

    import cupyx  # pylint: disable=import-outside-toplevel

    cp.scatter_add = cupyx.scatter_add

    spec = importlib.util.find_spec("optuna")
    if spec is None:
        logging.info("Optuna not installed, kernel opt is disabled")
        cp.optimize_kernels = contextlib.nullcontext
        cp.ExperimentalWarning = MockExperimentalWarning
    elif optimize:
        import optuna  # pylint: disable=import-outside-toplevel

        optuna.logging.set_verbosity(optuna.logging.WARN)
        logging.info("Using optuna to optimize kernels, the first calls will be slowwwww")
        cp.optimize_kernels = cupyx.optimizing.optimize
        cp.ExperimentalWarning = optuna.exceptions.ExperimentalWarning
    else:
        cp.optimize_kernels = contextlib.nullcontext
        cp.ExperimentalWarning = MockExperimentalWarning

    def cp_to_cpu(x, stream=None, out=None):
        """Take a np/cupy array and always return it in host memory (as an np array)."""
        if "cupy" in type(x).__module__:
            return x.get(stream=stream, out=out)
        if out is None:
            return x

        out[:] = x
        return out

    cp.to_cpu = cp_to_cpu

    # Default xp.array to fp32
    cp._oldarray = cp.array  # pylint: disable=protected-access

    def array_f32(*args, **kwargs):
        """replacement cp.array that forces all float64 allocs to be float32 instead."""
        ret = cp._oldarray(*args, **kwargs)  # pylint: disable=protected-access
        if ret.dtype == xp.float64:
            ret = ret.astype("float32")
        return ret

    cp.array = array_f32

    import cupyx.scipy.special  # pylint: disable=import-outside-toplevel

    cp.special = cupyx.scipy.special

    # grab pinned mem allocators
    cp.empty_pinned = cupyx.empty_pinned
    cp.empty_like_pinned = cupyx.empty_like_pinned
    cp.zeros_pinned = cupyx.zeros_pinned
    cp.zeros_like_pinned = cupyx.zeros_like_pinned

    bucky.xp = cp

    import cupyx.scipy.sparse as xp_sparse  # pylint: disable=import-outside-toplevel, redefined-outer-name

    bucky.xp_sparse = xp_sparse

    # TODO need to check cupy version is >9.0.0a1 in order to use sparse

    return 0
