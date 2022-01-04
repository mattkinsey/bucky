"""Class to read and store all the data from the bucky input graph."""
import datetime
from functools import partial

import networkx as nx
import pandas as pd
import yaml
from joblib import Memory

from ..numerical_libs import sync_numerical_libs, xp
from ..util.cached_prop import cached_property
from ..util.extrapolate import interp_extrap
from ..util.read_config import bucky_cfg
from ..util.rolling_mean import rolling_mean, rolling_window
from ..util.spline_smooth import fit
from .adjmat import buckyAij

memory = Memory(bucky_cfg["cache_dir"], verbose=0, mmap_mode="r")


@memory.cache
def cached_scatter_add(a, slices, value):
    ret = a.copy()
    xp.scatter_add(ret, slices, value)
    return ret


class buckyGraphData:
    """Contains and preprocesses all the data imported from an input graph file."""

    @sync_numerical_libs
    def clean_historical_data(self, cum_case_hist, cum_death_hist, inc_hosp, start_date, g_data, save_plots=True):

        # interpolate unreported days
        with open("included_data/state_update.yml", "r") as f:
            state_update_info = yaml.load(f, yaml.SafeLoader)  # nosec

        n_hist = cum_case_hist.shape[1]
        dow = xp.arange(start=-n_hist + start_date.weekday() + 1, stop=start_date.weekday() + 1, step=1)
        dow = dow % 7

        update_mask = xp.full_like(cum_case_hist, False, dtype=bool)
        default_state_mask = xp.isin(dow, xp.array(state_update_info[0]))

        for i in range(1, g_data.max_adm1 + 1):
            if i in state_update_info:
                state_mask = xp.isin(dow, xp.array(state_update_info[i]))
            else:
                state_mask = default_state_mask

            update_mask[g_data.adm1_id == i] = state_mask[None, ...]

        diff_mask_cases = xp.diff(cum_case_hist, axis=1, prepend=cum_case_hist[:, 0][..., None] - 1) != 0.0
        diff_mask_deaths = xp.diff(cum_death_hist, axis=1, prepend=cum_death_hist[:, 0][..., None] - 1) != 0.0
        # diff_mask = diff_mask_cases | diff_mask_deaths

        valid_case_mask = update_mask | diff_mask_cases
        valid_death_mask = update_mask | diff_mask_deaths

        new_cum_cases = xp.empty_like(cum_case_hist)
        new_cum_deaths = xp.empty_like(cum_case_hist)

        x = xp.arange(0, new_cum_cases.shape[1])
        for i in range(new_cum_cases.shape[0]):
            new_cum_cases[i] = interp_extrap(x, x[valid_case_mask[i]], cum_case_hist[i, valid_case_mask[i]])
            new_cum_deaths[i] = interp_extrap(x, x[valid_death_mask[i]], cum_death_hist[i, valid_death_mask[i]])

        # TODO remove massive outliers here, they lead to gibbs-like wiggling in the cumulative fitting

        # Apply spline smoothing
        df = max(1 * n_hist // 7 - 1, 4)
        # df = n_hist - n_hist//4
        alp = 1.5
        tol = 1.0e-5  # 6
        gam_inc = 8.0  # 8.
        gam_cum = 8.0  # 8.

        spline_cum_cases = xp.clip(
            fit(new_cum_cases, df=df, alp=alp, pirls=True, gamma=gam_cum, tol=tol, label="PIRLS Cumulative Cases"),
            a_min=0.0,
            a_max=None,
        )
        spline_cum_deaths = xp.clip(
            fit(new_cum_deaths, df=df, alp=alp, pirls=True, gamma=gam_cum, tol=tol, label="PIRLS Cumulative Deaths"),
            a_min=0.0,
            a_max=None,
        )

        # cum_rolling_cases = xp.mean(rolling_window(new_cum_cases, 7, center=True), axis=-1)
        # cum_rolling_deaths = xp.mean(rolling_window(new_cum_deaths, 7, center=True), axis=-1)

        # inc_spline_cases = xp.clip(xp.gradient(cum_rolling_cases, axis=1, edge_order=2), a_min=0.0, a_max=None)
        # inc_spline_deaths = xp.clip(xp.gradient(cum_rolling_deaths, axis=1, edge_order=2), a_min=0.0, a_max=None)

        # inc_rolling_cases = xp.mean(rolling_window(inc_spline_cases, 7, reflect_type="even", center=True), axis=-1)
        # inc_rolling_deaths = xp.mean(rolling_window(inc_spline_deaths, 7, reflect_type="even", center=True), axis=-1)

        inc_cases = xp.clip(xp.gradient(spline_cum_cases, axis=1, edge_order=2), a_min=0.0, a_max=None)
        inc_deaths = xp.clip(xp.gradient(spline_cum_deaths, axis=1, edge_order=2), a_min=0.0, a_max=None)

        for i in range(inc_cases.shape[0]):
            inc_cases[i] = interp_extrap(x, x[valid_case_mask[i]], inc_cases[i, valid_case_mask[i]])
            inc_deaths[i] = interp_extrap(x, x[valid_death_mask[i]], inc_deaths[i, valid_death_mask[i]])

        spline_inc_cases = xp.clip(
            fit(
                inc_cases,
                alp=alp,
                df=df - 1,
                dist="g",
                pirls=True,
                standardize=True,
                gamma=gam_inc,
                tol=tol,
                label="PIRLS Incident Cases",
            ),
            a_min=0.0,
            a_max=None,
        )
        spline_inc_deaths = xp.clip(
            fit(
                inc_deaths,
                alp=alp,
                df=df - 1,
                dist="g",
                pirls=True,
                standardize=True,
                gamma=1.5 * gam_inc,
                tol=tol,
                label="PIRLS Incident Deaths",
            ),
            a_min=0.0,
            a_max=None,
        )
        spline_inc_hosp = xp.clip(
            fit(
                inc_hosp,
                alp=alp,
                df=df - 1,
                dist="g",
                pirls=True,
                standardize=True,
                gamma=gam_inc,
                tol=tol,
                label="PIRLS Incident Hospitalizations",
            ),
            a_min=0.0,
            a_max=None,
        )

        # TODO cache this somehow b/c it takes awhile
        if save_plots:  # False:
            import matplotlib

            matplotlib.use("agg")
            import pathlib

            import matplotlib.pyplot as plt
            import tqdm
            import us

            from ..util.read_config import bucky_cfg

            # TODO we should drop these in raw_output_dir and have postprocess put them in the run's dir
            # TODO we could also drop the data for viz.plot...
            # if we just drop the data this should be moved to viz.historical_plots or something
            out_dir = bucky_cfg["output_dir"] + "/_historical_fit_plots"
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

            diff_cases = xp.diff(g_data.sum_adm1(cum_case_hist), axis=1)
            diff_deaths = xp.diff(g_data.sum_adm1(cum_death_hist), axis=1)

            fips_map = us.states.mapping("fips", "abbr")
            non_state_ind = xp.all(g_data.sum_adm1(cum_case_hist) < 1, axis=1)
            fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
            # TODO move the sum_adm1 calls out here, its doing that reduction ALOT
            for i in tqdm.tqdm(range(g_data.max_adm1 + 1), desc="Ploting fits", dynamic_ncols=True):
                if non_state_ind[i]:
                    continue
                fips_str = str(i).zfill(2)
                if fips_str in fips_map:
                    name = fips_map[fips_str] + " (" + fips_str + ")"
                else:
                    name = fips_str
                # fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
                ax = fig.subplots(nrows=2, ncols=4)
                ax[0, 0].plot(xp.to_cpu(g_data.sum_adm1(cum_case_hist)[i]), label="Cumulative Cases")
                ax[0, 0].plot(xp.to_cpu(g_data.sum_adm1(spline_cum_cases)[i]), label="Fit")
                ax[1, 0].plot(xp.to_cpu(g_data.sum_adm1(cum_death_hist)[i]), label="Cumulative Deaths")
                ax[1, 0].plot(xp.to_cpu(g_data.sum_adm1(spline_cum_deaths)[i]), label="Fit")
                ax[0, 1].plot(xp.to_cpu(diff_cases[i]), label="Incident Cases")
                ax[0, 1].plot(xp.to_cpu(g_data.sum_adm1(spline_inc_cases)[i]), label="Fit")
                ax[0, 2].plot(xp.to_cpu(diff_deaths[i]), label="Incident Deaths")
                ax[0, 2].plot(xp.to_cpu(g_data.sum_adm1(spline_inc_deaths)[i]), label="Fit")
                ax[1, 1].plot(xp.to_cpu(xp.log1p(diff_cases[i])), label="Log(Incident Cases)")
                ax[1, 1].plot(xp.to_cpu(xp.log1p(g_data.sum_adm1(spline_inc_cases)[i])), label="Fit")
                ax[1, 2].plot(xp.to_cpu(xp.log1p(diff_deaths[i])), label="Log(Incident Deaths)")
                ax[1, 2].plot(xp.to_cpu(xp.log1p(g_data.sum_adm1(spline_inc_deaths)[i])), label="Fit")
                ax[0, 3].plot(xp.to_cpu(inc_hosp[i]), label="Incident Hosp")
                ax[0, 3].plot(xp.to_cpu(spline_inc_hosp[i]), label="Fit")
                ax[1, 3].plot(xp.to_cpu(xp.log1p(inc_hosp[i])), label="Log(Incident Hosp)")
                ax[1, 3].plot(xp.to_cpu(xp.log1p(spline_inc_hosp[i])), label="Fit")

                log_cases = xp.to_cpu(xp.log1p(xp.clip(diff_cases[i], a_min=0.0, a_max=None)))
                log_deaths = xp.to_cpu(xp.log1p(xp.clip(diff_deaths[i], a_min=0.0, a_max=None)))
                if xp.any(xp.array(log_cases > 0)):
                    ax[1, 1].set_ylim([0.9 * xp.min(log_cases[log_cases > 0]), 1.1 * xp.max(log_cases)])
                if xp.any(xp.array(log_deaths > 0)):
                    ax[1, 2].set_ylim([0.9 * xp.min(log_deaths[log_deaths > 0]), 1.1 * xp.max(log_deaths)])

                ax[0, 0].legend()
                ax[1, 0].legend()
                ax[0, 1].legend()
                ax[1, 1].legend()
                ax[0, 2].legend()
                ax[1, 2].legend()
                ax[0, 3].legend()
                ax[1, 3].legend()

                fig.suptitle(name)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(out_dir + "/" + name + ".png")
                fig.clf()
            plt.close(fig)
            plt.close("all")

        return spline_cum_cases, spline_cum_deaths, spline_inc_cases, spline_inc_deaths, spline_inc_hosp

    @sync_numerical_libs
    def __init__(self, G, sparse=True, spline_smooth=True, force_diag_Aij=False):
        """Initialize the input data into cupy/numpy, reading it from a networkx graph."""

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
        self.n_hist = self.cum_case_hist.shape[0]
        self.Nij = _read_node_attr(G, "N_age_init", a_min=1.0)

        # TODO add adm0 to support multiple countries
        self.adm2_id = _read_node_attr(G, G.graph["adm2_key"], dtype=int)[0]
        self.adm1_id = _read_node_attr(G, G.graph["adm1_key"], dtype=int)[0]
        self.adm0_name = G.graph["adm0_name"]

        # in case we want to alloc something indexed by adm1/2
        self.max_adm2 = xp.to_cpu(xp.max(self.adm2_id))
        self.max_adm1 = xp.to_cpu(xp.max(self.adm1_id))

        self.Aij = buckyAij(G, sparse=sparse, a_min=0.0, force_diag=force_diag_Aij)

        self.have_hosp = "hhs_data" in G.graph
        if self.have_hosp:
            # adm1_current_hosp = xp.zeros((self.max_adm1 + 1,), dtype=float)
            hhs_data = G.graph["hhs_data"].reset_index()
            hhs_data["date"] = pd.to_datetime(hhs_data["date"])

            # ensure we're not cheating w/ future data
            hhs_data = hhs_data.loc[hhs_data.date <= pd.Timestamp(self.start_date)]

            # add int index for dates
            hhs_data["date_index"] = pd.Categorical(hhs_data.date, ordered=True).codes

            # sort to line up our indices
            hhs_data = hhs_data.set_index(["date_index", "adm1"]).sort_index()

            max_hhs_inds = hhs_data.index.max()
            adm1_current_hosp_hist = xp.zeros((max_hhs_inds[0] + 1, max_hhs_inds[1] + 1))
            adm1_inc_hosp_hist = xp.zeros((max_hhs_inds[0] + 1, max_hhs_inds[1] + 1))

            # collapse adult/pediatric columns
            tot_hosps = (
                hhs_data.total_adult_patients_hospitalized_confirmed_covid
                + hhs_data.total_pediatric_patients_hospitalized_confirmed_covid
            )
            inc_hosps = (
                hhs_data.previous_day_admission_adult_covid_confirmed
                + hhs_data.previous_day_admission_pediatric_covid_confirmed
            )

            # remove nans
            tot_hosps = tot_hosps.fillna(0.0)
            inc_hosps = inc_hosps.fillna(0.0)

            adm1_current_hosp_hist[tuple(xp.array(tot_hosps.index.to_list()).T)] = tot_hosps.to_numpy()
            adm1_inc_hosp_hist[tuple(xp.array(inc_hosps.index.to_list()).T)] = inc_hosps.to_numpy()

            # clip to same historical length as case/death data
            self.adm1_curr_hosp_hist = adm1_current_hosp_hist[-self.n_hist :]
            self.adm1_inc_hosp_hist = adm1_inc_hosp_hist[-self.n_hist :]

        # TODO move these params to config?
        if spline_smooth:
            self.rolling_mean_func_cum = partial(fit, df=self.cum_case_hist.shape[0] // 7)
        else:
            self._rolling_mean_type = "arithmetic"  # "geometric"
            self._rolling_mean_window_size = 7
            self.rolling_mean_func_cum = partial(
                rolling_mean,
                window_size=self._rolling_mean_window_size,
                axis=0,
                mean_type=self._rolling_mean_type,
            )

        # TODO remove the rolling mean stuff; it's deprecated
        self.rolling_mean_func_cum = lambda x, **kwargs: x

        (
            clean_cum_cases,
            clean_cum_deaths,
            clean_inc_cases,
            clean_inc_deaths,
            clean_inc_hosp,
        ) = self.clean_historical_data(
            self.cum_case_hist.T, self.cum_death_hist.T, self.adm1_inc_hosp_hist.T, self.start_date, self
        )
        self.cum_case_hist = clean_cum_cases.T
        self.cum_death_hist = clean_cum_deaths.T
        self.inc_case_hist = clean_inc_cases.T
        self.inc_death_hist = clean_inc_deaths.T
        self.adm1_inc_hosp_hist = clean_inc_hosp.T

    # TODO maybe provide a decorator or take a lambda or something to generalize it?
    # also this would be good if it supported rolling up to adm0 for multiple countries
    # memo so we don'y have to handle caching this on the input data?
    # TODO! this should be operating on last index, its super fragmented atm
    # also if we sort node indices by adm2 that will at least bring them all together...
    def sum_adm1(self, adm2_arr, mask=None, cache=False):
        """Return the adm1 sum of a variable defined at the adm2 level using the mapping on the graph."""
        # TODO add in axis param, we call this a bunch on array.T
        # assumes 1st dim is adm2 indexes
        # TODO should take an axis argument and handle reshape, then remove all the transposes floating around
        # TODO we should use xp.unique(return_inverse=True) to compress these rather than
        #  allocing all the adm1 ids that dont exist, see the new postprocess
        shp = (self.max_adm1 + 1,) + adm2_arr.shape[1:]
        out = xp.zeros(shp, dtype=adm2_arr.dtype)
        if mask is None:
            adm1_ids = self.adm1_id
        else:
            adm1_ids = self.adm1_id[mask]
            # adm2_arr = adm2_arr[mask]
        if cache:
            out = cached_scatter_add(out, adm1_ids, adm2_arr)
        else:
            xp.scatter_add(out, adm1_ids, adm2_arr)
        return out

    # TODO add scatter_adm2 with weights. Noone should need to check self.adm1/2_id outside this class

    # TODO other adm1 reductions (like harmonic mean), also add weights (for things like Nj)

    # Define and cache some of the reductions on Nij we might want
    @cached_property
    def Nj(self):
        """Total population per adm2."""
        return xp.sum(self.Nij, axis=0)

    @cached_property
    def N(self):
        """Total population."""
        return xp.sum(self.Nij)

    @cached_property
    def adm0_Ni(self):
        """Age stratified adm0 population."""
        return xp.sum(self.Nij, axis=1)

    @cached_property
    def adm1_Nij(self):
        """Age stratified adm1 populations."""
        return self.sum_adm1(self.Nij.T).T

    @cached_property
    def adm1_Nj(self):
        """Total adm1 populations."""
        return self.sum_adm1(self.Nj)

    # Define and cache some rolling means of the historical data @ adm2
    @cached_property
    def rolling_inc_cases(self):
        """Return the rolling mean of incident cases."""
        return xp.clip(
            self.rolling_mean_func_cum(
                xp.clip(xp.gradient(self.cum_case_hist, axis=0, edge_order=2), a_min=0, a_max=None),
            ),
            a_min=0.0,
            a_max=None,
        )

    @cached_property
    def rolling_inc_deaths(self):
        """Return the rolling mean of incident deaths."""
        return xp.clip(
            self.rolling_mean_func_cum(
                xp.clip(xp.gradient(self.cum_death_hist, axis=0, edge_order=2), a_min=0, a_max=None),
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

    @cached_property
    def rolling_adm1_curr_hosp(self):
        """Return the rolling mean of cumulative deaths."""
        return xp.clip(self.rolling_mean_func_cum(self.adm1_curr_hosp_hist), a_min=0.0, a_max=None)

    @cached_property
    def rolling_adm1_inc_hosp(self):
        """Return the rolling mean of cumulative deaths."""
        return xp.clip(self.rolling_mean_func_cum(self.adm1_inc_hosp_hist), a_min=0.0, a_max=None)

    # adm1 rollups of historical data
    @cached_property
    def adm1_cum_case_hist(self):
        """Cumulative cases by adm1."""
        return self.sum_adm1(self.cum_case_hist.T).T

    @cached_property
    def adm1_inc_case_hist(self):
        """Incident cases by adm1."""
        return self.sum_adm1(self.inc_case_hist.T).T

    @cached_property
    def adm1_cum_death_hist(self):
        """Cumulative deaths by adm1."""
        return self.sum_adm1(self.cum_death_hist.T).T

    @cached_property
    def adm1_inc_death_hist(self):
        """Incident deaths by adm1."""
        return self.sum_adm1(self.inc_death_hist.T).T

    # adm0 rollups of historical data
    @cached_property
    def adm0_cum_case_hist(self):
        """Cumulative cases at adm0."""
        return xp.sum(self.cum_case_hist, axis=1)

    @cached_property
    def adm0_inc_case_hist(self):
        """Incident cases at adm0."""
        return xp.sum(self.inc_case_hist, axis=1)

    @cached_property
    def adm0_cum_death_hist(self):
        """Cumulative deaths at adm0."""
        return xp.sum(self.cum_death_hist, axis=1)

    @cached_property
    def adm0_inc_death_hist(self):
        """Incident deaths at adm0."""
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
