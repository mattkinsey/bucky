"""Class to read and store all the data from the bucky input graph."""
import datetime
import logging
import warnings
from functools import partial

import networkx as nx
import pandas as pd
from joblib import Memory
from numpy import RankWarning

from ..numerical_libs import sync_numerical_libs, xp
from ..util.cached_prop import cached_property
from ..util.extrapolate import interp_extrap
from ..util.power_transforms import YeoJohnson
from ..util.read_config import bucky_cfg
from ..util.rolling_mean import rolling_mean, rolling_window
from ..util.spline_smooth import fit, lin_reg
from .adjmat import buckyAij

memory = Memory(bucky_cfg["cache_dir"], verbose=0, mmap_mode="r")


@memory.cache
def cached_scatter_add(a, slices, value):
    """scatter_add() thats cached by joblib."""
    ret = a.copy()
    xp.scatter_add(ret, slices, value)
    return ret


class buckyGraphData:
    """Contains and preprocesses all the data imported from an input graph file."""

    # pylint: disable=too-many-public-methods

    @staticmethod
    @sync_numerical_libs
    def clean_historical_data(cum_case_hist, cum_death_hist, inc_hosp, start_date, g_data, force_save_plots=False):
        """Preprocess the historical data to smooth it and remove outliers."""
        n_hist = cum_case_hist.shape[1]

        adm1_case_hist = g_data.sum_adm1(cum_case_hist)
        adm1_death_hist = g_data.sum_adm1(cum_death_hist)
        adm1_diff_mask_cases = (
            xp.around(xp.diff(adm1_case_hist, axis=1, prepend=adm1_case_hist[:, 0][..., None]), 2) >= 1.0
        )
        adm1_diff_mask_death = (
            xp.around(xp.diff(adm1_death_hist, axis=1, prepend=adm1_death_hist[:, 0][..., None]), 2) >= 1.0
        )

        adm1_enough_case_data = (adm1_case_hist[:, -1] - adm1_case_hist[:, 0]) > n_hist
        adm1_enough_death_data = (adm1_death_hist[:, -1] - adm1_death_hist[:, 0]) > n_hist

        adm1_enough_data = adm1_enough_case_data | adm1_enough_death_data

        valid_adm1_mask = adm1_diff_mask_cases | adm1_diff_mask_death

        valid_adm1_case_mask = valid_adm1_mask
        valid_adm1_death_mask = valid_adm1_mask

        for i in range(adm1_case_hist.shape[0]):
            data = adm1_case_hist[i]
            rw = rolling_window(data, 3, center=True)
            mask = xp.around(xp.abs((data - xp.mean(lin_reg(rw, return_fit=True), axis=1)) / data), 2) < 0.1
            valid_adm1_case_mask[i] = valid_adm1_mask[i] & mask

        valid_case_mask = valid_adm1_case_mask[g_data.adm1_id]
        valid_death_mask = valid_adm1_death_mask[g_data.adm1_id]
        enough_data = adm1_enough_data[g_data.adm1_id]

        new_cum_cases = xp.empty_like(cum_case_hist)
        new_cum_deaths = xp.empty_like(cum_case_hist)

        x = xp.arange(0, new_cum_cases.shape[1])
        for i in range(new_cum_cases.shape[0]):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    if ~enough_data[i]:
                        new_cum_cases[i] = cum_case_hist[i]
                        new_cum_deaths[i] = cum_death_hist[i]
                        continue
                    new_cum_cases[i] = interp_extrap(
                        x,
                        x[valid_case_mask[i]],
                        cum_case_hist[i, valid_case_mask[i]],
                        n_pts=7,
                        order=2,
                    )
                    new_cum_deaths[i] = interp_extrap(
                        x,
                        x[valid_death_mask[i]],
                        cum_death_hist[i, valid_death_mask[i]],
                        n_pts=7,
                        order=2,
                    )
            except (TypeError, RankWarning, ValueError) as e:
                logging.error(e)

        # TODO remove massive outliers here, they lead to gibbs-like wiggling in the cumulative fitting

        new_cum_cases = xp.around(new_cum_cases, 6) + 0.0  # plus zero to convert -0 to 0.
        new_cum_deaths = xp.around(new_cum_deaths, 6) + 0.0

        # Apply spline smoothing
        df = max(1 * n_hist // 7 - 1, 4)

        alp = 1.5
        tol = 1.0e-5  # 6
        gam_inc = 8.0  # 2.4  # 8.
        gam_cum = 8.0  # 2.4  # 8.

        # df2 = int(10 * n_hist ** (2.0 / 9.0)) + 1  # from gam book section 4.1.7
        gam_inc = 2.4  # 2.4
        gam_cum = 2.4  # 2.4
        # tol = 1e-3

        spline_cum_cases = xp.clip(
            fit(
                new_cum_cases,
                df=df,
                alp=alp,
                gamma=gam_cum,
                tol=tol,
                label="PIRLS Cumulative Cases",
                standardize=False,
            ),
            a_min=0.0,
            a_max=None,
        )
        spline_cum_deaths = xp.clip(
            fit(
                new_cum_deaths,
                df=df,
                alp=alp,
                gamma=gam_cum,
                tol=tol,
                label="PIRLS Cumulative Deaths",
                standardize=False,
            ),
            a_min=0.0,
            a_max=None,
        )

        inc_cases = xp.clip(xp.gradient(spline_cum_cases, axis=1, edge_order=2), a_min=0.0, a_max=None)
        inc_deaths = xp.clip(xp.gradient(spline_cum_deaths, axis=1, edge_order=2), a_min=0.0, a_max=None)

        inc_cases = xp.around(inc_cases, 6) + 0.0
        inc_deaths = xp.around(inc_deaths, 6) + 0.0
        inc_hosp = xp.around(inc_hosp, 6) + 0.0

        # power_transform1 = BoxCox()
        # power_transform2 = BoxCox()
        # power_transform3 = BoxCox()
        power_transform1 = YeoJohnson()
        power_transform2 = YeoJohnson()
        power_transform3 = YeoJohnson()
        # Need to clip negatives for BoxCox
        # inc_cases = xp.clip(inc_cases, a_min=0., a_max=None)
        # inc_deaths = xp.clip(inc_deaths, a_min=0., a_max=None)
        # inc_hosp = xp.clip(inc_hosp, a_min=0., a_max=None)
        inc_cases = power_transform1.fit(inc_cases)
        inc_deaths = power_transform2.fit(inc_deaths)
        inc_hosp2 = power_transform3.fit(inc_hosp)

        inc_cases = xp.around(inc_cases, 6) + 0.0
        inc_deaths = xp.around(inc_deaths, 6) + 0.0
        inc_hosp = xp.around(inc_hosp, 6) + 0.0

        inc_fit_args = {
            "alp": alp,
            "df": df,  # df // 2 + 2 - 1,
            "dist": "g",
            "standardize": False,  # True,
            "gamma": gam_inc,
            "tol": tol,
            "clip": (0.0, None),
            "bootstrap": False,  # True,
        }

        all_cached = (
            fit.check_call_in_cache(inc_cases, **inc_fit_args)
            and fit.check_call_in_cache(inc_deaths, **inc_fit_args)
            and fit.check_call_in_cache(inc_hosp2, **inc_fit_args)
        )

        spline_inc_cases = fit(
            inc_cases,
            **inc_fit_args,
            label="PIRLS Incident Cases",
        )
        spline_inc_deaths = fit(
            inc_deaths,
            **inc_fit_args,
            label="PIRLS Incident Deaths",
        )
        spline_inc_hosp = fit(
            inc_hosp2,
            **inc_fit_args,
            label="PIRLS Incident Hospitalizations",
        )
        for _ in range(5):
            resid = spline_inc_cases - inc_cases
            stddev = xp.quantile(xp.abs(resid), axis=1, q=0.682)
            clean_resid = xp.clip(resid / (6.0 * stddev[:, None] + 1e-8), -1.0, 1.0)
            robust_weights = xp.clip(1.0 - clean_resid ** 2.0, 0.0, 1.0) ** 2.0
            spline_inc_cases = fit(inc_cases, **inc_fit_args, label="PIRLS Incident Cases", w=robust_weights)

            resid = spline_inc_deaths - inc_deaths
            stddev = xp.quantile(xp.abs(resid), axis=1, q=0.682)
            clean_resid = xp.clip(resid / (6.0 * stddev[:, None] + 1e-8), -1.0, 1.0)
            robust_weights = xp.clip(1.0 - clean_resid ** 2.0, 0.0, 1.0) ** 2.0
            spline_inc_deaths = fit(inc_deaths, **inc_fit_args, label="PIRLS Incident Deaths", w=robust_weights)

            resid = spline_inc_hosp - inc_hosp2
            stddev = xp.quantile(xp.abs(resid), axis=1, q=0.682)
            clean_resid = xp.clip(resid / (6.0 * stddev[:, None] + 1e-8), -1.0, 1.0)
            robust_weights = xp.clip(1.0 - clean_resid ** 2.0, 0.0, 1.0) ** 2.0
            spline_inc_hosp = fit(inc_hosp2, **inc_fit_args, label="PIRLS Incident Hosps", w=robust_weights)

        spline_inc_cases = power_transform1.inv(spline_inc_cases)
        spline_inc_deaths = power_transform2.inv(spline_inc_deaths)
        spline_inc_hosp = power_transform3.inv(spline_inc_hosp)

        # Only plot if the fits arent in the cache already
        # TODO this wont update if doing a historical run thats already cached
        save_plots = (not all_cached) or force_save_plots

        if save_plots:
            # pylint: disable=import-outside-toplevel
            import matplotlib

            matplotlib.use("agg")
            import pathlib

            import matplotlib.pyplot as plt
            import numpy as np
            import tqdm
            import us

            # TODO we should drop these in raw_output_dir and have postprocess put them in the run's dir
            # TODO we could also drop the data for viz.plot...
            # if we just drop the data this should be moved to viz.historical_plots or something
            out_dir = pathlib.Path(bucky_cfg["output_dir"]) / "_historical_fit_plots"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_dir.touch(exist_ok=True)  # update mtime

            diff_cases = xp.diff(g_data.sum_adm1(cum_case_hist), axis=1)
            diff_deaths = xp.diff(g_data.sum_adm1(cum_death_hist), axis=1)

            fips_map = us.states.mapping("fips", "abbr")
            non_state_ind = xp.all(g_data.sum_adm1(cum_case_hist) < 1, axis=1)

            fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
            x = xp.arange(cum_case_hist.shape[1])
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

                ax[0, 0].fill_between(
                    xp.to_cpu(x),
                    xp.to_cpu(xp.min(adm1_case_hist[i])),
                    xp.to_cpu(xp.max(adm1_case_hist[i])),
                    where=xp.to_cpu(~valid_adm1_case_mask[i]),
                    color="grey",
                    alpha=0.2,
                )
                ax[1, 0].plot(xp.to_cpu(g_data.sum_adm1(cum_death_hist)[i]), label="Cumulative Deaths")
                ax[1, 0].plot(xp.to_cpu(g_data.sum_adm1(spline_cum_deaths)[i]), label="Fit")
                ax[1, 0].fill_between(
                    xp.to_cpu(x),
                    xp.to_cpu(xp.min(adm1_death_hist[i])),
                    xp.to_cpu(xp.max(adm1_death_hist[i])),
                    where=xp.to_cpu(~valid_adm1_death_mask[i]),
                    color="grey",
                    alpha=0.2,
                )

                ax[0, 1].plot(xp.to_cpu(diff_cases[i]), label="Incident Cases")
                ax[0, 1].plot(xp.to_cpu(g_data.sum_adm1(spline_inc_cases)[i]), label="Fit")
                ax[0, 1].fill_between(
                    xp.to_cpu(x),
                    xp.to_cpu(xp.min(diff_cases[i])),
                    xp.to_cpu(xp.max(diff_cases[i])),
                    where=xp.to_cpu(~valid_adm1_case_mask[i]),
                    color="grey",
                    alpha=0.2,
                )

                ax[0, 2].plot(xp.to_cpu(diff_deaths[i]), label="Incident Deaths")
                ax[0, 2].plot(xp.to_cpu(g_data.sum_adm1(spline_inc_deaths)[i]), label="Fit")
                ax[0, 2].fill_between(
                    xp.to_cpu(x),
                    xp.to_cpu(xp.min(diff_deaths[i])),
                    xp.to_cpu(xp.max(diff_deaths[i])),
                    where=xp.to_cpu(~valid_adm1_death_mask[i]),
                    color="grey",
                    alpha=0.2,
                )

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
                plt.savefig(out_dir / (name + ".png"))
                fig.clf()
            plt.close(fig)
            plt.close("all")

            spline_inc_hosp_adm2 = (
                spline_inc_hosp[g_data.adm1_id] * (g_data.Nj / g_data.adm1_Nj[g_data.adm1_id])[:, None]
            )
            df = {
                "cum_cases_fitted": spline_cum_cases,
                "cum_deaths_fitted": spline_cum_deaths,
                "inc_cases_fitted": spline_inc_cases,
                "inc_deaths_fitted": spline_inc_deaths,
                "inc_hosp_fitted": spline_inc_hosp_adm2,
            }
            df["adm2"] = xp.broadcast_to(g_data.adm2_id[:, None], spline_cum_cases.shape)
            df["adm1"] = xp.broadcast_to(g_data.adm1_id[:, None], spline_cum_cases.shape)
            dates = [str(start_date + datetime.timedelta(days=int(i))) for i in np.arange(-n_hist + 1, 1)]
            df["date"] = np.broadcast_to(np.array(dates)[None, :], spline_cum_cases.shape)
            for k in df:
                df[k] = xp.ravel(xp.to_cpu(df[k]))

            # TODO sort columns
            df = pd.DataFrame(df)
            df.to_csv(out_dir / "fit_data.csv", index=False)

            # embed()

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

        self.Aij = buckyAij(G, sparse=sparse, edge_a_min=0.0, force_diag=force_diag_Aij)

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
            self.cum_case_hist.T,
            self.cum_death_hist.T,
            self.adm1_inc_hosp_hist.T,
            self.start_date,
            self,
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
