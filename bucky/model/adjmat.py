""" Utility class to manage the adjacency matrix regardless of if its dense or sparse."""
# import logging

# import networkx as nx

from ..numerical_libs import sync_numerical_libs, xp, xp_sparse
from ..util.distributions import truncnorm


# TODO sprase diag will be added to cupy v9.0.0a2 so we won't need to track it after that
# but there still might be some performance to be gained by tracking it and only having to calc diag once...
class buckyAij:
    """Class that handles the adjacency matrix for the model, generalizes between dense/sparse."""

    @sync_numerical_libs
    def __init__(self, G, sparse=True, a_min=0.0, force_diag=False):
        """Initialize the stored matrix off of the edges of a networkx graph."""

        self.sparse = sparse
        # TODO handle different attr names for edge weight

        if force_diag:
            # TODO this assumes sparse == True
            self._base_Aij = xp_sparse.identity(self._base_Aij.shape[0], format="csr")
            self._base_Aij_diag = xp.ones(self._base_Aij.shape[0])
        else:
            self._base_Aij, self._base_Aij_diag = _read_edge_mat(G, sparse=sparse, a_min=a_min)

        if self.sparse:
            self._indptr_sorted = _csr_is_ind_sorted(self._base_Aij)

        self._Aij, self._Aij_diag = self.normalize(self._base_Aij, self._base_Aij_diag, axis=0)

    def normalize(self, mat, mat_diag, axis=0):
        """Normalize A along a given axis and keep the cache A_diag in sync."""
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
        """property refering to the dense/sparse matrix."""
        return (self._Aij + xp_sparse.eye(m=self._Aij.shape[0], format="csr")) / 2.0

    @property
    def diag(self):
        """property refering to the cache diagional of the matrix."""
        return (self._Aij_diag + 1.0) / 2.0

    def perturb(self, var):
        """Apply a normal perturbation to the matrix (and keep its diag in sync)."""
        # Roll for perturbation in shape of Aij
        # we have to get tricky here b/c of lots of missing cupy methods
        if self.sparse:
            fac_shp = self._base_Aij.data.shape
        else:
            fac_shp = self._base_Aij.shape

        fac = truncnorm(1.0, var, fac_shp, a_min=1e-6)

        # rescale Aij from base_Aij
        if self.sparse:
            self._Aij.data = self._base_Aij.data * fac
            # TODO if scipy.sparse just use their diagnoal()?
            # _csr_diag(self._Aij, out=self._Aij_diag, indptr_sorted=self._indptr_sorted)
            self._Aij_diag = self._Aij.diagonal()
        else:
            self._Aij = self._base_Aij * fac
            self._Aij_diag = self._Aij.diagonal()

        self._Aij, self._Aij_diag = self.normalize(self._Aij, self._Aij_diag, axis=0)


def _read_edge_mat(G, weight_attr="weight", sparse=True, a_min=0.0):
    """Read the adj matrix of a networkx graph and convert it to the cupy/scipy format."""
    edges = xp.array(list(G.edges(data=weight_attr))).T
    # TODO fix: edges = edges.T[edges[2] <= a_min].T  # clip edges with weight < a_min
    A = xp_sparse.coo_matrix((edges[2], (edges[0].astype(int), edges[1].astype(int))))
    A = A.tocsr()  # just b/c it will do this for almost every op on the array anyway...
    if not sparse:
        A = A.toarray()
    A_diag = edges[2][edges[0] == edges[1]]
    return A, A_diag


def _csr_diag(mat, out=None, indptr_sorted=False):
    """Get the diagonal of a scipy/cupy CSR sparse matrix quickly.

    .. deprecated:: 0.9.0
            As of cupy==9.0.0a2 this functionality is provided by scipy.sparse.csr.csr_matrix.diagonal()
    """
    if ~indptr_sorted:
        raise NotImplementedError
        # we'd just have to replace searchsorted with argmax but it will go from O(logN) to O(N)
        # or just sort the indices
    if out is None:
        out = xp.empty(mat.shape[0])
    for i in range(out.size):
        row_inds = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        diag_ind = mat.indptr[i] + xp.searchsorted(row_inds, i - 1)
        out[i] = mat.data[diag_ind]

    return out


def _csr_is_ind_sorted(mat):
    """Check if a cupy/scipy CSR sparse matrix has its indices sorted."""
    out = xp.zeros(mat.shape[0])
    for i in range(mat.shape[0]):
        row_inds = mat.indices[mat.indptr[i] : mat.indptr[i + 1]]
        out[i] = xp.all(xp.diff(row_inds) > 0)
    return xp.all(out)
