from loguru import logger

from ..numerical_libs import sync_numerical_libs, xp, xp_sparse
from .array_utils import rolling_window
from .power_transforms import YeoJohnson
from .spline_smooth import lin_reg, ridge


def gaussian_kernel(x, tau):
    x_outer_diff = x[:, :, None] - x[:, None, :]
    x_normed_outer_diff = x_outer_diff / xp.max(x_outer_diff, axis=(1, 2), keepdims=True)
    numer = xp.einsum("ijk,ikj->ijk", x_normed_outer_diff, x_normed_outer_diff)
    w = xp.exp(numer / (2 * tau * tau))
    return w


def tricubic_kernel(x, f=1.0 / 8.0):
    x_outer_diff = x[:, :, None] - x[:, None, :]
    x_normed_outer_diff = x_outer_diff / xp.max(f * x_outer_diff, axis=(1, 2), keepdims=True)
    w = (1.0 - xp.abs(x_normed_outer_diff) ** 3) ** 3
    return xp.clip(w, a_min=0.0, a_max=None)


@sync_numerical_libs
def loess(y, x=None, tau=0.05, iters=1, degree=2, q=None, data_w=None):

    if x is None:
        x = xp.arange(0, y.shape[1], dtype=float)
        x = xp.tile(x, (y.shape[0], 1))

    y_min = xp.min(y, axis=1, keepdims=True)
    y_max = xp.max(y, axis=1, keepdims=True)
    y_range = y_max - y_min + 1e-8
    y_norm = (y - y_min) / y_range

    if q is not None:
        # TODO move to tricuibic_kernel function
        h = [xp.sort(xp.abs(x - x[:, i][:, None]), axis=1)[:, q] for i in range(x.shape[1])]
        h = xp.stack(h).T
        w = xp.clip(xp.abs((x[:, :, None] - x[:, None, :]) / h[..., None]), a_min=0.0, a_max=1.0)
        w = (1.0 - w**3) ** 3
    else:
        w = gaussian_kernel(x, tau)

    if data_w is not None:
        iters = 1
        delta = data_w
    else:
        delta = xp.ones_like(y)

    if iters > 1:
        logger.error("Robust weight for LOESS is still WIP!")
        raise NotImplementedError

    out = xp.empty_like(y)

    for _ in range(iters):

        # TODO we can get rid of the for loop with a reshape into the batch dim...
        for i in range(y.shape[1]):
            out[:, i] = lin_reg(y_norm, x, w=delta * w[:, i], alp=0.6, degree=degree)[:, i]

        res = y - out
        # stddev = xp.quantile(xp.abs(res), axis=1, q=0.682)
        res_star = res / (6.0 * xp.median(xp.abs(res), axis=1, keepdims=True) + 1.0e-8)
        clean_resid = res_star
        # clean_resid = xp.clip(res / (6.0 * stddev[:, None] + 1e-8), -1.0, 1.0)
        robust_weights = xp.clip(1.0 - clean_resid**2.0, 0.0, 1.0) ** 2.0
        delta = robust_weights

    out = out * y_range + y_min

    return out
