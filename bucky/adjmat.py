import logging

import networkx as nx

from .util.distributions import truncnorm

xp = None
xp_sparse = None

# TODO add a min value to init and clip it off to make things more sparse


class buckyAij:
    def __init__(self, G, sparse=True):
        global xp, xp_sparse
        if xp is None:
            from . import xp
        if xp_sparse is None:
            from . import xp_sparse

        self.sparse = sparse
        self._base_Aij, self._base_Aij_diag = _read_edge_mat(G, sparse=sparse)
        if self.sparse:
            self._indptr_sorted = _csr_is_ind_sorted(self._base_Aij)

        self._Aij, self._Aij_diag = self.normalize(self._base_Aij, self._base_Aij_diag, axis=0)

    def normalize(self, mat, mat_diag, axis=0):
        mat_norm_fac = 1.0 / mat.sum(axis=axis)
        mat_norm_fac = xp.array(mat_norm_fac)
        if self.sparse:
            mat = mat.multiply(mat_norm_fac).tocsr()
        else:
            mat = mat * mat_norm_fac

        if mat_diag is not None:
            mat_diag = xp.squeeze(xp.multiply(mat_diag, mat_norm_fac))

        return mat, mat_diag

    @property
    def A(self):
        return self._Aij

    @property
    def diag(self):
        return self._Aij_diag

    def perturb(self, var):
        # Roll for perturbation in shape of Aij
        # we have to get tricky here b/c of lots of missing cupy methods
        if self.sparse:
            fac_shp = self._base_Aij.data.shape
        else:
            fac_shp = self._base_Aij.shape

        fac = truncnorm(xp, 1.0, var, fac_shp, a_min=1e-6)

        # rescale Aij from base_Aij
        if self.sparse:
            self._Aij.data = self._base_Aij.data * fac
            # TODO if scipy.sparse just use their diagnoal()?
            _csr_diag(self._Aij, out=self._Aij_diag, indptr_sorted=self._indptr_sorted)
        else:
            self._Aij = self._base_Aij * fac
            self._Aij_diag = self._Aij.diagonal()

        self._Aij, self._Aij_diag = self.normalize(self._Aij, self._Aij_diag, axis=0)


def _read_edge_mat(G, weight_attr="weight", sparse=True):
    edges = xp.array(list(G.edges(data=weight_attr))).T
    A = xp_sparse.coo_matrix((edges[2], (edges[0].astype(int), edges[1].astype(int))))
    A = A.tocsr()  # just b/c it will do this for almost every op on the array anyway...
    if not sparse:
        A = A.toarray()
    A_diag = edges[2][edges[0] == edges[1]]
    return A, A_diag


def _csr_diag(mat, out=None, indptr_sorted=False):
    if ~indptr_sorted:
        raise NotImplementedError
        # we'd just have to replace searchsorted with argmax but it will go from O(logN) to O(N)
    if out is None:
        out = xp.empty(mat.shape[0])
    for i in range(out.size):
        row_inds = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        diag_ind = mat.indptr[i] + xp.searchsorted(row_inds, i - 1)
        out[i] = mat.data[diag_ind]

    return out


def _csr_is_ind_sorted(mat):
    out = xp.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        row_inds = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        out[i] = xp.all(xp.diff(row_inds) > 0)
    return xp.all(out)
