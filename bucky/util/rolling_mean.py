"""Provides generic rolling Pythagorean means over cupy/numpy ndarrays."""
from ..numerical_libs import sync_numerical_libs, xp


@sync_numerical_libs
def rolling_mean(arr, window_size=7, axis=0, weights=None, mean_type="arithmetic"):
    """Calculate a rolling mean over a numpy/cupy ndarray."""
    # we could probably just pass args/kwargs...
    if mean_type == "arithmetic":
        return _rolling_arithmetic_mean(arr, window_size, axis, weights)
    elif mean_type == "geometric":
        return _rolling_geometric_mean(arr, window_size, axis, weights)
    elif mean_type == "harmonic":  # noqa: SIM106
        return _rolling_harmonic_mean(arr, window_size, axis, weights)
    else:
        raise RuntimeError  # TODO what type of err should go here?


@sync_numerical_libs
def rolling_window(a, window_size, center=True, axis=0, pad=True, pad_mode="reflect", reflect_type="odd", freq=1):
    """Use stride_tricks to add an extra dim on the end of an ndarray for each elements window."""

    if pad:
        pad_before = xp.zeros(len(a.shape), dtype=xp.int32)
        pad_after = xp.zeros(len(a.shape), dtype=xp.int32)
        if center:
            # only allow odd sized centered windows
            if not (window_size % 2):
                raise ValueError
            pad_size = window_size // 2
            pad_before[axis] = pad_size
            pad_after[axis] = pad_size
        else:
            pad_before[axis] = window_size - 1

        padding = list(zip(list(xp.to_cpu(pad_before)), list(xp.to_cpu(pad_after))))
        a = xp.pad(a, padding, mode=pad_mode, reflect_type=reflect_type)

    shape = list(a.shape)
    shape[axis] = a.shape[axis] - window_size + 1
    shape = tuple(shape) + (window_size,)
    strides = a.strides + (a.strides[axis],)
    freq_inds = [slice(None)] * (axis) + [slice(0, shape[axis], freq)] + [slice(None)] * (a.ndim - axis - 1)
    return xp.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[tuple(freq_inds)]


def _rolling_arithmetic_mean(arr, window_size=7, axis=0, weights=None):
    """Compute a rolling arithmetic mean."""
    arr = xp.swapaxes(arr, axis, -1)

    if weights is None:
        rolling_arr = xp.mean(rolling_window(arr, window_size), axis=-1)
    else:
        window = weights / xp.sum(weights) * window_size
        window = xp.broadcast_to(window, arr.shape + (window_size,))
        rolling_arr = xp.mean(window * rolling_window(arr, window_size), axis=-1)

    rolling_arr = xp.swapaxes(rolling_arr, axis, -1)
    return rolling_arr


def _rolling_geometric_mean(arr, window_size, axis=0, weights=None):
    """Compute a rolling geometric mean."""
    # TODO add some error checking (for negatives, etc)
    arr = xp.swapaxes(arr, axis, -1)

    if weights is None:
        rolling_arr = xp.exp(xp.nanmean(rolling_window(xp.log(arr), window_size), axis=-1))
    else:
        window = weights / xp.sum(weights) * window_size
        window = xp.broadcast_to(window, arr.shape + (window_size,))
        rolling_arr = xp.exp(xp.nanmean(window * rolling_window(xp.log(arr), window_size), axis=-1))

    rolling_arr = xp.swapaxes(rolling_arr, axis, -1)
    return rolling_arr


def _rolling_harmonic_mean(arr, window_size, axis=0, weights=None):
    """Compute a rolling harmonic mean."""
    # TODO check for 0s
    arr = xp.swapaxes(arr, axis, -1).astype(float)

    if weights is None:
        rolling_arr = xp.reciprocal(xp.nanmean(rolling_window(xp.reciprocal(arr), window_size), axis=-1))
    else:
        window = weights / xp.sum(weights) * window_size
        window = xp.broadcast_to(window, arr.shape + (window_size,))
        rolling_arr = xp.reciprocal(xp.nanmean(window * rolling_window(xp.reciprocal(arr), window_size), axis=-1))

    rolling_arr = xp.swapaxes(rolling_arr, axis, -1)
    return rolling_arr
