"""Utility class to manage the adjacency matrix regardless of if its dense or sparse."""
import logging
import operator
from functools import reduce

from ..numerical_libs import sync_numerical_libs, xp, xp_sparse
from ..util.distributions import truncnorm

# TODO Generalize this to include Cij too, mainly perturb/normalize


class buckyAij:
    """Class that handles the adjacency matrix for the model, generalizes between dense/sparse."""

    @sync_numerical_libs
    def __init__(self, G, weight_attr="weight", force_diag=False, sparse_format="csr", n_nodes=None):
        """Initialize the stored matrix off of the edges of a networkx graph."""

        # init the array as sparse so memory doesnt blow up (just in case)
        self.sparse = True
        self.sparse_format = sparse_format

        if ~force_diag:
            raise NotImplementedError
        else:
            # cupy is still missing a bunch of dia format functionality :(
            # self.sparse_format = "dia"
            if G is not None:
                size = G.number_of_nodes()
            else:
                size = n_nodes
            self._base_Aij = xp_sparse.identity(size, format=sparse_format)

        # if sparsity < .5? just automatically make it dense?
        # same if it's fairly small? (<100 rows?)

        self._Aij = self._normalize(self._base_Aij, axis=0)
        logging.info(f"Loaded Aij: size={self._base_Aij.shape}, sparse={self.sparse}, format={self.sparse_format}")

    def todense(self):
        """Convert to dense."""
        self.sparse = False
        self.sparse_format = None
        self._base_Aij = self._base_Aij.toarray()
        self._Aij = self._Aij.toarray()

    def tosparse(self, sparse_format="csr"):
        """Convert to sparse."""
        self.sparse = True
        self.sparse_format = sparse_format
        self._base_Aij = self._base_Aij.asformat(self.sparse_format)
        self._Aij = self._Aij.asformat(self.sparse_format)

    def _normalize(self, mat, axis=0):
        """Normalize A along a given axis."""

        mat_norm_fac = 1.0 / mat.sum(axis=axis)
        if self.sparse:
            mat = mat.multiply(mat_norm_fac).asformat(self.sparse_format)
            mat.eliminate_zeros()
        else:
            mat = mat * mat_norm_fac

        return mat

    @property
    def sparsity(self):
        """Return the sparsity of the matrix."""
        # NB: can't use .count_nonzero() because it might cast to dense first?
        # it's also way slower as long as we don't have explict zeros
        n_tot = float(reduce(operator.mul, self._base_Aij.shape))
        if self.sparse:
            return 1.0 - self._base_Aij.getnnz() / n_tot
        else:
            return xp.sum(self._base_Aij == 0.0) / n_tot

    @property
    def A(self):
        """Property refering to the dense/sparse matrix."""
        return self._Aij

    @property
    def diag(self):
        """Property refering to the cache diagional of the matrix."""
        return self._Aij.diagonal()

    def perturb(self, var):
        """Apply a normal perturbation to the matrix (and keep its diag in sync)."""
        # Roll for perturbation in shape of Aij
        if self.sparse:
            fac_shp = self._base_Aij.data.shape
        else:
            fac_shp = self._base_Aij.shape

        fac = truncnorm(1.0, var, fac_shp, a_min=1e-6)

        # rescale Aij from base_Aij
        if self.sparse:
            self._Aij.data = self._base_Aij.data * fac
        else:
            self._Aij = self._base_Aij * fac

        self._Aij = self._normalize(self._Aij, axis=0)
