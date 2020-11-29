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

import numpy as xp
import scipy.integrate._ivp.ivp as xp_ivp
import scipy.sparse as xp_sparse
import scipy.special

# Default imports for cpu code
# This will be overwritten with a call to .numerical_libs.use_cupy()
import bucky

xp.scatter_add = xp.add.at
xp.optimize_kernels = contextlib.nullcontext
xp.to_cpu = lambda x, **kwargs: x  # one arg noop
xp.special = scipy.special

bucky.xp = xp
bucky.xp_sparse = xp_sparse
bucky.xp_ivp = xp_ivp


class MockExperimentalWarning(Warning):
    """Simple class to mock the optuna warning if we don't have optuna."""


xp.ExperimentalWarning = MockExperimentalWarning


def reimport_numerical_libs():
    """Reimport xp, xp_sparse, xp_ivp from the global context (in case they've been update to cupy)."""
    for lib in ("xp", "xp_sparse", "xp_ivp"):
        caller_globals = dict(inspect.getmembers(inspect.stack()[1][0]))["f_globals"]
        if lib in caller_globals:
            bucky_module = importlib.import_module("bucky")
            caller_globals[lib] = getattr(bucky_module, lib)


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
    import numpy as np  # pylint: disable=import-outside-toplevel, reimported

    cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.memory.malloc_managed).malloc)

    # add cupy search sorted for scipy.ivp (this was only added to cupy sometime between v6.0.0 and v7.0.0)
    if ~hasattr(cp, "searchsorted"):
        # NB: this isn't correct in general but it works for what scipy solve_ivp needs...
        def cp_searchsorted(a, v, side="right", sorter=None):
            """Provide a cupy version of search sorted thats good enough for scipy.ivp

            This was added to cupy sometime between v6.0.0 and v7.0.0 so it won't be needed for up to date installs.

            .. warning:: This isn't correct in general but it works for what scipy.ivp needs..
            """
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
        bucky.xp_ivp = modify_and_import(
            "scipy.integrate._ivp." + name,
            None,
            lambda src: src.replace("import numpy", "import cupy"),
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
        return x

    cp.to_cpu = cp_to_cpu

    # Add a version of np.r_ to cupy that just calls numpy
    # has to be a class b/c r_ uses sq brackets
    class cp_r_:  # pylint: disable=too-few-public-methods
        """Hackish version of a cupy version of r_.

        It just uses numpy and wraps it to have the same signature.
        """  # noqa: RST306

        def __getitem__(self, inds):
            """Call np.r_ and case the result to cupy."""  # noqa: RST306
            return cp.array(np.r_[inds])

    cp.r_ = cp_r_()

    import cupyx.scipy.special  # pylint: disable=import-outside-toplevel

    cp.special = cupyx.scipy.special

    bucky.xp = cp

    import cupyx.scipy.sparse as xp_sparse  # pylint: disable=import-outside-toplevel, redefined-outer-name

    bucky.xp_sparse = xp_sparse

    # TODO need to check cupy version is >9.0.0a1 in order to use sparse

    return 0
