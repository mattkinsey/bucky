"""Provides various scoring metrics for probabilistic forecasts."""
from ..numerical_libs import sync_numerical_libs, xp


def logistic(x, x0=0.0, k=1.0, L=1.0):
    """Logistic function.

    Parameters
    ----------
    x:
    x0:
    k:
    L:
    Returns
    -------
    ndarray
    """
    return L / (1.0 + xp.exp(-k * x - x0))


def IS(x, lower, upper, alp):
    """Interval score.

    Parameters
    ----------
    x:
    lower:
    upper:
    alp:
    Returns
    -------
    ndarray
    """
    return (upper - lower) + 2.0 / alp * (lower - x) * (x < lower) + 2.0 / alp * (x - upper) * (x > upper)


def smooth_IS(x, lower, upper, alp):
    """Approx Interval Score with smooth derivatives.

    Parameters
    ----------
    x:
    lower:
    upper:
    alp:
    Returns
    -------
    ndarray
    """
    width = upper - lower
    return (
        (upper - lower)
        + 2.0 / alp * (lower - x) * logistic(-x, x0=-lower, k=2.0 / width)
        + 2.0 / alp * (x - upper) * logistic(x, x0=upper, k=2.0 / width)
    )


@sync_numerical_libs
def WIS(x, q, x_q, norm=False, log=False, smooth=False):
    """Weighted Interval Score.

    Parameters
    ----------
    x:
    q:
    x_q:
    norm: bool, optional
    log: bool, optional
    smooth: bool, optional
    Returns
    -------
    ndarray
    """
    # todo sort q and x_q based on q
    K = len(q) // 2
    alps = xp.array([1 - q[-i - 1] + q[i] for i in range(K)])
    Fs = xp.array([[x_q[i], x_q[-i - 1]] for i in range(K)])
    m = x_q[K + 1]
    w0 = 0.5
    wk = alps / 2.0

    if smooth:
        ret = 1.0 / (K + 1.0) * (w0 * 2 * xp.abs(x - m) + xp.sum(wk * smooth_IS(x, Fs[:, 0], Fs[:, 1], alps)))
    else:
        ret = (
            1.0
            / (K + 1.0)
            * (
                w0 * 2 * xp.abs(x - m)
                + xp.sum(wk[..., None] * IS(x[None, ...], Fs[:, 0], Fs[:, 1], alps[..., None]), axis=0)
            )
        )
    if norm:
        mask = xp.nonzero(x)
        ret[mask] = ret[mask] / (x[mask] + 1.0)
    if log:
        ret = xp.log1p(ret)
    return ret
