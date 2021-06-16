"""Class to read and store all the data from the bucky input graph."""
from functools import partial

import networkx as nx

from ..numerical_libs import sync_numerical_libs, xp
from ..util.cached_prop import cached_property
from ..util.rolling_mean import rolling_mean, rolling_window
from ..util.spline_smooth import fit
from .adjmat import buckyAij


class buckyGraphData:
    """Contains and preprocesses all the data imported from an input graph file."""

    @sync_numerical_libs
    def __init__(self, G, sparse=True, spline_smooth=False):
        """Initialize the input data into cupy/numpy, reading it from a networkx graph"""

        # make sure G is sorted by adm2 id
        adm2_ids = _read_node_attr(G, G.graph["adm2_key"], dtype=int)[0]
        if ~(xp.diff(adm2_ids) >= 0).all():  # this is pretty much std::is_sorted without the errorchecking
            H = nx.DiGraph()
            H.add_nodes_from(sorted(G.nodes(data=True), key=lambda node: node[1][G.graph["adm2_key"]]))
            H.add_edges_from(G.edges(data=True))
            H.graph = G.graph
            G = H.copy()

        G = nx.convert_node_labels_to_integers(G)
        self.start_date = datetime.date.fromisoformat(G.graph["start_date"])
        self.cum_case_hist, self.inc_case_hist = _read_node_attr(G, "case_hist", diff=True, a_min=0.0)
        self.cum_death_hist, self.inc_death_hist = _read_node_attr(G, "death_hist", diff=True, a_min=0.0)
        self.Nij = _read_node_attr(G, "N_age_init", a_min=1.0)

        # Perform some outlier detection/correction on the cases/deaths
        # TODO this should be a cleaned up and made a utility function for general timeseries
        old_cases = self.inc_case_hist.copy()
        old_deaths = self.inc_death_hist.copy()
        old_ccases = self.cum_case_hist.copy()
        old_cdeaths = self.cum_death_hist.copy()
        # clean up incidence data to remove data dumps
        for arrs in ((self.inc_case_hist, self.cum_case_hist),):  # (self.inc_death_hist, self.cum_death_hist)):
            tmp = arrs[0].T
            cum_arr = arrs[1]
            rmedian = xp.median(rolling_window(tmp, 7), axis=-1)
            rstd = xp.var(rolling_window(tmp, 7), axis=-1)
            rmean = xp.mean(rolling_window(tmp, 7), axis=-1)
            mask = xp.abs(tmp - rmedian) > 5.0 * (rmedian + rmean) / 2.0
            mask = mask.T
            rmedian = rmedian.T
            tmp = arrs[0].copy()
            arrs[0][mask] = rmedian[mask]
            cum_change = xp.cumsum(tmp - arrs[0], axis=0)
            arrs[1][:] = cum_arr - cum_change

        # TODO add adm0 to support multiple countries
        self.adm2_id = _read_node_attr(G, G.graph["adm2_key"], dtype=int)[0]
        self.adm1_id = _read_node_attr(G, G.graph["adm1_key"], dtype=int)[0]
        self.adm0_name = G.graph["adm0_name"]

        # in case we want to alloc something indexed by adm1/2
        self.max_adm2 = xp.to_cpu(xp.max(self.adm2_id))
        self.max_adm1 = xp.to_cpu(xp.max(self.adm1_id))

        self.Aij = buckyAij(G, sparse, a_min=0.0)

        # TODO move these params to config?
        if spline_smooth:
            self.rolling_mean_func_cum = partial(fit, df=self.cum_case_hist.shape[1] // 7)
        else:
            self._rolling_mean_type = "arithmetic"  # "geometric"
            self._rolling_mean_window_size = 7
            self.rolling_mean_func_cum = partial(
                rolling_mean,
                window_size=self._rolling_mean_window_size,
                axis=0,
                mean_type=self._rolling_mean_type,
            )

    # TODO maybe provide a decorator or take a lambda or something to generalize it?
    # also this would be good if it supported rolling up to adm0 for multiple countries
    # memo so we don'y have to handle caching this on the input data?
    # TODO! this should be operating on last index, its super fragmented atm
    # also if we sort node indices by adm2 that will at least bring them all together...
    def sum_adm1(self, adm2_arr, mask=None):
        """Return the adm1 sum of a variable defined at the adm2 level using the mapping on the graph."""
        # TODO add in axis param, we call this a bunch on array.T
        # assumes 1st dim is adm2 indexes
        # TODO should take an axis argument and handle reshape, then remove all the transposes floating around
        # TODO we should use xp.unique(return_inverse=True) to compress these rather than allocing all the adm1 ids that dont exist, see the new postprocess
        shp = (self.max_adm1 + 1,) + adm2_arr.shape[1:]
        out = xp.zeros(shp, dtype=adm2_arr.dtype)
        if mask is None:
            adm1_ids = self.adm1_id
        else:
            adm1_ids = self.adm1_id[mask]
            # adm2_arr = adm2_arr[mask]
        xp.scatter_add(out, adm1_ids, adm2_arr)
        return out

    # TODO add scatter_adm2 with weights. Noone should need to check self.adm1/2_id outside this class

    # TODO other adm1 reductions (like harmonic mean), also add weights (for things like Nj)

    # Define and cache some of the reductions on Nij we might want
    @cached_property
    def Nj(self):
        """Total population per adm2"""
        return xp.sum(self.Nij, axis=0)

    @cached_property
    def N(self):
        """Total population"""
        return xp.sum(self.Nij)

    @cached_property
    def adm0_Ni(self):
        """Age stratified adm0 population"""
        return xp.sum(self.Nij, axis=1)

    @cached_property
    def adm1_Nij(self):
        """Age stratified adm1 populations"""
        return self.sum_adm1(self.Nij.T).T

    @cached_property
    def adm1_Nj(self):
        """Total adm1 populations"""
        return self.sum_adm1(self.Nj)

    # Define and cache some rolling means of the historical data @ adm2
    @cached_property
    def rolling_inc_cases(self):
        """Return the rolling mean of incident cases."""
        return xp.clip(
            self.rolling_mean_func_cum(
                xp.clip(xp.gradient(self.cum_case_hist, axis=0, edge_order=2), a_min=0, a_max=None)
            ),
            a_min=0.0,
            a_max=None,
        )

    @cached_property
    def rolling_inc_deaths(self):
        """Return the rolling mean of incident deaths."""
        return xp.clip(
            self.rolling_mean_func_cum(
                xp.clip(xp.gradient(self.cum_death_hist, axis=0, edge_order=2), a_min=0, a_max=None)
            ),
            a_min=0.0,
            a_max=None,
        )

    @cached_property
    def rolling_cum_cases(self):
        """Return the rolling mean of cumulative cases."""
        return xp.clip(self.rolling_mean_func_cum(self.cum_case_hist), a_min=0.0, a_max=None)

    @cached_property
    def rolling_cum_deaths(self):
        """Return the rolling mean of cumulative deaths."""
        return xp.clip(self.rolling_mean_func_cum(self.cum_death_hist), a_min=0.0, a_max=None)

    # adm1 rollups of historical data
    @cached_property
    def adm1_cum_case_hist(self):
        """Cumulative cases by adm1"""
        return self.sum_adm1(self.cum_case_hist.T).T

    @cached_property
    def adm1_inc_case_hist(self):
        """Incident cases by adm1"""
        return self.sum_adm1(self.inc_case_hist.T).T

    @cached_property
    def adm1_cum_death_hist(self):
        """Cumulative deaths by adm1"""
        return self.sum_adm1(self.cum_death_hist.T).T

    @cached_property
    def adm1_inc_death_hist(self):
        """Incident deaths by adm1"""
        return self.sum_adm1(self.inc_death_hist.T).T

    # adm0 rollups of historical data
    @cached_property
    def adm0_cum_case_hist(self):
        """Cumulative cases at adm0"""
        return xp.sum(self.cum_case_hist, axis=1)

    @cached_property
    def adm0_inc_case_hist(self):
        """Incident cases at adm0"""
        return xp.sum(self.inc_case_hist, axis=1)

    @cached_property
    def adm0_cum_death_hist(self):
        """Cumulative deaths at adm0"""
        return xp.sum(self.cum_death_hist, axis=1)

    @cached_property
    def adm0_inc_death_hist(self):
        """Incident deaths at adm0"""
        return xp.sum(self.inc_death_hist, axis=1)


def _read_node_attr(G, name, diff=False, dtype=xp.float32, a_min=None, a_max=None):
    """Read an attribute from every node into a cupy/numpy array and optionally clip and/or diff it."""
    clipping = (a_min is not None) or (a_max is not None)
    node_list = list(nx.get_node_attributes(G, name).values())
    arr = xp.vstack(node_list).astype(dtype).T
    if clipping:
        arr = xp.clip(arr, a_min=a_min, a_max=a_max)

    if diff:
        arr_diff = xp.gradient(arr, axis=0, edge_order=2).astype(dtype)
        if clipping:
            arr_diff = xp.clip(arr_diff, a_min=a_min, a_max=a_max)
        return arr, arr_diff

    return arr
