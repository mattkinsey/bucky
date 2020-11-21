""" Provides generic rolling Pythagorean means over cupy/numpy ndarrays"""
from ..numerical_libs import reimport_numerical_libs, xp


def rolling_mean(arr, window_size=7, axis=0, weights=None, mean_type="arithmetic"):
    reimport_numerical_libs()
    # we could probably just pass args/kwargs...
    if mean_type == "arithmetic":
        return _rolling_arithmetic_mean(arr, window_size, axis, weights)
    elif mean_type == "geometric":
        return _rolling_geometric_mean(arr, window_size, axis, weights)
    elif mean_type == "harmonic":
        return _rolling_harmonic_mean(arr, window_size, axis, weights)
    else:
        raise RuntimeError  # TODO what type of err should go here?


def _rolling_arithmetic_mean(arr, window_size=7, axis=0, weights=None):
    arr = xp.swapaxes(arr, axis, -1)
    shp = arr.shape[:-1] + (arr.shape[-1] - window_size + 1,)
    rolling_arr = xp.empty(shp)
    if weights is None:
        window = xp.ones(window_size) / window_size
    else:
        window = weights / xp.sum(weights)
    arr = arr.reshape(-1, arr.shape[-1])
    for i in range(arr.shape[0]):  # we can use stride_tricks to speed this up if needed
        rolling_arr[i] = xp.convolve(arr[i], window, mode="valid")
        rolling_arr = rolling_arr.reshape(shp)
    rolling_arr = xp.swapaxes(rolling_arr, axis, -1)
    return rolling_arr


def _rolling_geometric_mean(arr, window_size, axis=0, weights=None):
    # add support for weights (need to use a log identity)
    if weights is not None:
        raise NotImplementedError
    arr = xp.swapaxes(arr, axis, -1)
    shp = arr.shape[:-1] + (arr.shape[-1] - window_size + 1,)
    rolling_arr = xp.empty(shp)
    window = xp.ones(window_size) / window_size
    arr = arr.reshape(-1, arr.shape[-1])
    log_abs_arr = xp.log(xp.abs(arr))
    neg_mask = arr < 0.0
    log_abs_arr[xp.abs(arr) < 1.0] = -1000.0
    for i in range(arr.shape[0]):
        tmp = xp.convolve(log_abs_arr[i], window, mode="valid")
        n_neg = xp.convolve(1.0 * neg_mask[i], xp.ones(window_size), mode="valid")
        # n_neg = xp.sum(arr[i] < 0.)
        rolling_arr[i] = ((-1.0) ** n_neg) ** (1.0 / window_size) * xp.exp(tmp)
    rolling_arr = rolling_arr.reshape(shp)
    rolling_arr = xp.swapaxes(rolling_arr, axis, -1)
    return rolling_arr


def _rolling_harmonic_mean(arr, window_size, axis=0, weights=None):
    raise NotImplementedError
