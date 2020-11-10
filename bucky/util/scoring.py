"""Provides various scoring metrics for probabilistic forecasts.
"""
import numpy as np


def logistic(x, x0=0.0, k=1.0, L=1.0):
    """TODO: Summary

    Parameters
    ----------
    x : TYPE
        TODO: Description
    x0 : float, optional
        TODO: Description
    k : float, optional
        TODO: Description
    L : float, optional
        TODO: Description

    Returns
    -------
    :
        TODO

    Deleted Parameters
    ------------------
    :
        TODO
    """
    return L / (1.0 + np.exp(-k * x - x0))


def IS(x, lower, upper, alp):
    """TODO: Summary

    Parameters
    ----------
    x : TYPE
        TODO: Description
    lower : TYPE
        TODO: Description
    upper : TYPE
        TODO: Description
    alp : TYPE
        TODO: Description

    Returns
    -------
    :
        TODO

    Deleted Parameters
    ------------------
    :
        TODO
    """
    return (upper - lower) + 2.0 / alp * (lower - x) * (x < lower) + 2.0 / alp * (x - upper) * (x > upper)


def smooth_IS(x, lower, upper, alp):
    """TODO: Summary

    Parameters
    ----------
    x : TYPE
        TODO: Description
    lower : TYPE
        TODO: Description
    upper : TYPE
        TODO: Description
    alp : TYPE
        TODO: Description

    Returns
    -------
    :
        TODO

    Deleted Parameters
    ------------------
    :
        TODO
    """
    width = upper - lower
    return (
        (upper - lower)
        + 2.0 / alp * (lower - x) * logistic(-x, x0=-lower, k=2.0 / width)
        + 2.0 / alp * (x - upper) * logistic(x, x0=upper, k=2.0 / width)
    )


def WIS(x, q, x_q, norm=False, log=False, smooth=False):
    """
    TODO: Summary

    Parameters
    ----------
    x : TYPE
        TODO: Description
    q : TYPE
        TODO: Description
    x_q : TYPE
        TODO: Description
    norm : bool, optional
        TODO: Description
    log : bool, optional
        TODO: Description
    smooth : bool, optional
        TODO: Description

    Returns
    -------
    :
        TODO

    Deleted Parameters
    ------------------
    :
        TODO
    """
    # todo sort q and x_q based on q
    K = len(q) // 2
    alps = np.array([1 - q[-i - 1] + q[i] for i in range(K)])
    Fs = np.array([[x_q[i], x_q[-i - 1]] for i in range(K)])
    m = x_q[K + 1]
    w0 = 0.5
    wk = alps / 2.0
    if smooth:
        ret = 1.0 / (K + 1.0) * (w0 * 2 * np.abs(x - m) + np.sum(wk * smooth_IS(x, Fs[:, 0], Fs[:, 1], alps)))
    else:
        ret = 1.0 / (K + 1.0) * (w0 * 2 * np.abs(x - m) + np.sum(wk * IS(x, Fs[:, 0], Fs[:, 1], alps)))
    if norm:
        ret /= x
    if log:
        ret = np.log(ret)
    return ret
