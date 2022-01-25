"""Utility class to manage the adjacency matrix regardless of if its dense or sparse."""
import logging
import operator
from functools import reduce

from ..numerical_libs import sync_numerical_libs, xp, xp_sparse
from ..util.distributions import truncnorm


class buckyAij:
    """Class that handles the adjacency matrix for the model, generalizes between dense/sparse."""

    @sync_numerical_libs
    def __init__(self, G, weight_attr="weight", edge_a_min=0.0, force_diag=False, sparse_format="csr"):
        """Initialize the stored matrix off of the edges of a networkx graph."""

        # init the array as sparse so memory doesnt blow up (just in case)
        self.sparse = True
        self.sparse_format = sparse_format

        if force_diag:
            # cupy is still missing a bunch of dia format functionality :(
            # self.sparse_format = "dia"
            self._base_Aij = xp_sparse.identity(G.number_of_nodes(), format=sparse_format)
        else:
            self._base_Aij = self._read_nx_edge_mat(G, weight_attr=weight_attr, edge_a_min=edge_a_min)

        # if sparsity < .5? just automatically make it dense?
        # same if it's fairly small? (<100 rows?)

        self._Aij = self.normalize(self._base_Aij, axis=0)
        logging.info(f"Loaded Aij: size={self._base_Aij.shape}, sparse={self.sparse}, format={self.sparse_format}")

    def todense(self):
        """Convert to dense."""
        self.sparse = False
        self.sparse_format = None
        self._base_Aij = self._base_Aij.toarray()

    def tosparse(self, sparse_format="csr"):
        """Convert to sparse."""
        self.sparse = True
        self.sparse_format = sparse_format
        self._base_Aij = self._base_Aij.asformat(self.sparse_format)

    def normalize(self, mat, axis=0):
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

        self._Aij = self.normalize(self._Aij, axis=0)

    def _read_nx_edge_mat(self, G, weight_attr="weight", edge_a_min=0.0):
        """Read the adj matrix of a networkx graph and convert it to the cupy/scipy sparse format."""
        # pylint: disable=unused-argument
        edges = xp.array(list(G.edges(data=weight_attr))).T
        # TODO fix: edges = edges.T[edges[2] <= a_min].T  # clip edges with weight < a_min
        # could also do this clipping based of quantiles (i.e. remove 30% of weakest edges)
        A = xp_sparse.coo_matrix((edges[2], (edges[0].astype(int), edges[1].astype(int))))
        A = A.asformat(self.sparse_format)
        A.eliminate_zeros()
        return A
