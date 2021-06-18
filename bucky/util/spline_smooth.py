"""Rough method of smoothing data w/ splines. Based off the implementation of cr() in patsy."""
from collections import defaultdict

from ..numerical_libs import sync_numerical_libs, xp

dtype = xp.float32


def _get_natural_f(knots):
    """Returns mapping of natural cubic spline values to 2nd derivatives."""
    h = knots[:, 1:] - knots[:, :-1]
    diag = (h[:, :-1] + h[:, 1:]) / 3.0
    ul_diag = h[:, 1:-1] / 6.0

    d = xp.zeros((knots.shape[0], knots.shape[1] - 2, knots.shape[1]), dtype=dtype)
    for i in range(knots.shape[1] - 2):
        d[:, i, i] = 1.0 / h[:, i]
        d[:, i, i + 2] = 1.0 / h[:, i + 1]
        d[:, i, i + 1] = -d[:, i, i] - d[:, i, i + 2]

    A = xp.zeros((knots.shape[0], knots.shape[1] - 2, knots.shape[1] - 2), dtype=dtype)
    for i in range(knots.shape[0]):
        A[i] += xp.diag(ul_diag[i], -1) + xp.diag(diag[i]) + xp.diag(ul_diag[i], 1)

    fm = xp.linalg.solve(A, d)

    # fm = linalg.solve_banded((1, 1), banded_b, d)

    return xp.hstack([xp.zeros((knots.shape[0], 1, knots.shape[1])), fm, xp.zeros((knots.shape[0], 1, knots.shape[1]))])


def _find_knots_lower_bounds(x, knots):
    """Find the lower bound for the knots."""
    lb = xp.empty(x.shape, dtype=int)
    for i in range(knots.shape[0]):
        lb[i] = xp.searchsorted(knots[i], x[i]) - 1
    lb[lb == -1] = 0
    lb[lb == knots.shape[1] - 1] = knots.shape[1] - 2

    return lb


def _compute_base_functions(x, knots):
    """Return base functions for the spline basis."""
    j = _find_knots_lower_bounds(x, knots)

    h = knots[:, 1:] - knots[:, :-1]
    hj = xp.take_along_axis(h, j, axis=1)
    xj1_x = xp.take_along_axis(knots, j + 1, axis=1) - x
    x_xj = x - xp.take_along_axis(knots, j, axis=1)

    ajm = xj1_x / hj
    ajp = x_xj / hj

    cjm_3 = xj1_x * xj1_x * xj1_x / (6.0 * hj)
    cjm_3[x > xp.max(knots, axis=1, keepdims=True)] = 0.0
    cjm_1 = hj * xj1_x / 6.0
    cjm = cjm_3 - cjm_1

    cjp_3 = x_xj * x_xj * x_xj / (6.0 * hj)
    cjm_3[x < xp.min(knots, axis=1, keepdims=True)] = 0.0
    cjp_1 = hj * x_xj / 6.0
    cjp = cjp_3 - cjp_1

    return ajm, ajp, cjm, cjp, j


def nunique(arr, axis=-1):
    """Return the number of uniq values along a given axis."""
    arr_sorted = xp.sort(arr, axis=axis)
    n_not_uniq = (xp.diff(arr_sorted, axis=axis) == 0).sum(1)
    return arr.shape[axis] - n_not_uniq


def _get_free_crs_dmatrix(x, knots):
    """Builds an unconstrained cubic regression spline design matrix."""
    knots_dict = {}  # defaultdict(list)
    x_knots_dict_map = {}  # defaultdict(list)

    # find the uniques sets of knots so we don't do alot of redundant work
    # batch_u_knots, u_knots_x_map = xp.unique(knots, return_inverse=True, axis=0)
    """
    n_knots = nunique(batch_u_knots, axis=-1)
    uniq_n_knots = xp.unique(n_knots)
    #x_map = 

    for n in uniq_n_knots:
        # indices of batch_u_knots that have n knots
        n_inds = xp.argwhere(n_knots == n)[:,0]
        # get all the knots w/ n uniq vals
        batch_n_knots = batch_u_knots[n_inds]
        # isolate uniq knots so size is (..., n)
        knots_dict[n] = xp.vstack([xp.unique(k, axis=-1) for k in batch_n_knots])
        x_knots_dict_map[n] = n_inds

    embed()
    """

    # enforce uniqueness of knots for set of knots still in x

    knots_dict = defaultdict(list)
    knots_dict_map = defaultdict(list)
    for i, k in enumerate(knots):
        u_knots = xp.unique(k)
        knots_dict[u_knots.size].append(u_knots)
        knots_dict_map[u_knots.size].append(i)
        # embed()

    for n_knots in knots_dict:
        knots_dict[n_knots] = xp.stack(knots_dict[n_knots])
        knots_dict_map[n_knots] = xp.array(knots_dict_map[n_knots])

    # handle each basis with n knots seperately
    dm_dict = {}
    for n in knots_dict:
        if n < 3:
            continue
        ajm, ajp, cjm, cjp, j = _compute_base_functions(x[knots_dict_map[n]], knots_dict[n])
        j1 = j + 1
        f = _get_natural_f(knots_dict[n])
        # dmt = ajm * i[j, :].T + ajp * i[j1, :].T + cjm * f[j, :].T + cjp * f[j1, :].T

        eye = xp.identity(n, dtype=dtype)
        if True:
            # if we're using cupy we cant batch it b/c it will build an intermediate array in mem that is HUGE
            dm = xp.empty(x[knots_dict_map[n]].shape + (n,), dtype=dtype)
            for i in range(dm.shape[0]):
                dm[i] = xp.einsum("j,jk->jk", ajm[i], eye[j[i], :])
                dm[i] += xp.einsum("j,jk->jk", ajp[i], eye[j1[i], :])
                dm[i] += xp.einsum("j,jk->jk", cjm[i], f[i][j[i], :])
                dm[i] += xp.einsum("j,jk->jk", cjp[i], f[i][j1[i], :])
        else:
            dm = xp.einsum("ij,ijk->ijk", ajm, eye[j, :])
            dm += xp.einsum("ij,ijk->ijk", ajp, eye[j1, :])
            dm += xp.einsum("ij,iijk->ijk", cjm, f[:, j, :])
            dm += xp.einsum("ij,iijk->ijk", cjp, f[:, j1, :])
        dm_dict[n] = dm

    return dm_dict, knots_dict_map


def _absorb_constraints(design_matrix, constraints):
    """Apply constraints to the design matrix."""
    m = constraints.shape[1]
    ret = xp.empty((design_matrix.shape[0], design_matrix.shape[1], design_matrix.shape[2] - m))
    # Have to do this one by one for now
    # batched qr solver for cupy issue: https://github.com/cupy/cupy/issues/4986
    for i in range(constraints.shape[0]):
        q, r = xp.linalg.qr(xp.transpose(constraints[i]), mode="complete")
        ret[i] = xp.dot(design_matrix[i], q[:, m:])

    return ret


def _cr(x, df, center=True):
    """Python version of the R lib mgcv function cr()."""

    # TODO make df settable to a vector
    n_constraints = 0
    if center:
        n_constraints = 1

    n_inner_knots = df - 2 + n_constraints

    # embed()
    # _get_all_sorted_knots from patsy
    # TODO add lower_bound param, well need to mask those values out the x array too
    lower_bound = xp.min(x, axis=-1)
    upper_bound = xp.max(x, axis=-1)
    inner_knots_q = xp.linspace(0, 100, n_inner_knots + 2, dtype=dtype)[1:-1]
    inner_knots = xp.asarray(xp.percentile(x, inner_knots_q, axis=-1))
    # all_knots = xp.concatenate(([lower_bound, upper_bound], inner_knots))
    # all_knots = xp.concatenate((xp.atleast_1d(lower_bound), inner_knots, xp.atleast_1d(upper_bound)))
    all_knots = xp.vstack([lower_bound, inner_knots, upper_bound]).T
    # all_knots = xp.unique(all_knots)

    dm_dict, dict_x_knot_map = _get_free_crs_dmatrix(x, all_knots)
    if center:
        for n in dm_dict:
            constraint = dm_dict[n].mean(axis=1).reshape((dm_dict[n].shape[0], 1, dm_dict[n].shape[2]))
            dm_dict[n] = _absorb_constraints(dm_dict[n], constraint)

    return dm_dict, dict_x_knot_map


def ridge(x, y, alp=0.0):
    """Calculate the exact soln to the ridge regression of the weights for basis x that fit data y."""
    # xtx = xp.dot(x.T, x)
    xtx = xp.einsum("ijk,ijl->ikl", x, x)
    # t1 = xp.empty(xtx.shape)
    # for i in range(x.shape[0]):
    #    t1[i] = xp.linalg.inv(alp*xp.identity(xtx.shape[1]) + xtx[i])
    # embed()
    t1 = xp.linalg.inv(alp * xp.tile(xp.identity(xtx.shape[1]), (xtx.shape[0], 1, 1)) + xtx)

    # t2 = xp.dot(x.T, y)
    t2 = xp.einsum("ijk,ij->ik", x, y)
    # w = xp.dot(t1,t2)
    w = xp.einsum("ijk,ij->ik", t1, t2)
    return w


@sync_numerical_libs
def fit(y, x=None, df=10, alp=0.6):
    """Perform fit of natural cubic splines to the vector y, return the smoothed y."""
    # TODO handle df and alp as vectors

    # standardize ixputs
    if x is None:
        x = xp.arange(0, y.shape[1])
        x = xp.tile(x, (y.shape[0], 1))
    x_mean = xp.mean(x, axis=1, keepdims=True)
    x_var = xp.var(x, axis=1, keepdims=True)
    x_in = (x - x_mean) / (x_var + 1e-10)
    y_mean = xp.mean(y, axis=1, keepdims=True)
    y_var = xp.var(y, axis=1, keepdims=True)
    y_in = (y - y_mean) / (y_var + 1e-10)

    bs_dict, x_map = _cr(x_in, df=df)
    y_fit = xp.empty(y_in.shape)
    for n in bs_dict:
        bs = bs_dict[n]
        full_bs = xp.dstack([xp.ones((bs.shape[0], bs.shape[1], 1)), bs])
        coefs = ridge(full_bs, y_in[x_map[n]], alp=alp)
        y_fit[x_map[n]] = xp.sum((coefs[:, None, :] * full_bs), axis=-1)

    # rescale the standaridized output
    y_out = y_fit * y_var + y_mean

    # embed()

    return y_out
