"""The main module handling the simulation"""
import copy
import datetime
import logging
import os
import pickle
import random
import sys
import warnings
from pprint import pformat  # TODO set some defaults for width/etc with partial?

import numpy as np
import tqdm

from ..numerical_libs import enable_cupy, reimport_numerical_libs, xp, xp_ivp
from ..util.distributions import approx_mPERT
from ..util.fractional_slice import frac_last_n_vals
from ..util.util import TqdmLoggingHandler, _banner, get_runid
from .arg_parser_model import parser
from .estimation import estimate_cfr, estimate_chr, estimate_crr, estimate_Rt
from .exceptions import SimulationException
from .graph import buckyGraphData
from .io import BuckyOutputWriter
from .mc_instance import buckyMCInstance
from .npi import get_npi_params
from .optimize import test_opt
from .parameters import buckyParams
from .rhs import RHS_func
from .state import buckyState
from .vacc import buckyVaccAlloc

SCENARIO_HUB = False  # True

logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
)


class buckyModelCovid:
    """Class that handles one full simulation (both time integration and managing MC states)"""

    def __init__(
        self,
        debug=False,
        sparse_aij=False,
        t_max=None,
        graph_file=None,
        par_file=None,
        npi_file=None,
        disable_npi=False,
        reject_runs=False,
        output_dir=None,
    ):
        """Initialize the class, do some bookkeeping and read in the input graph"""
        self.debug = debug
        self.sparse = sparse_aij  # we can default to none and autodetect
        # w/ override (maybe when #adm2 > 5k and some sparsity critera?)

        # Integrator params
        self.t_max = t_max
        self.run_id = get_runid()
        logging.info(f"Run ID: {self.run_id}")

        self.npi_file = npi_file
        self.disable_npi = disable_npi
        self.reject_runs = reject_runs

        self.output_dates = None

        # COVID/model params from par file
        self.bucky_params = buckyParams(par_file)
        self.consts = self.bucky_params.consts
        # self.flags = self.bucky_params.flags # TODO split off bool consts into flags

        self.g_data = self.load_graph(graph_file)

        self.writer = BuckyOutputWriter(output_dir, self.run_id)
        self.writer.write_metadata(self.g_data, self.t_max)

    def update_params(self, update_dict):
        """Update the params based of a dict of new values"""
        self.bucky_params.update_params(update_dict)
        self.consts = self.bucky_params.consts

    def load_graph(self, graph_file):
        """Load the graph data and calculate all the variables that are static across MC runs"""
        # TODO refactor to just have this return g_data

        logging.info("loading graph")
        with open(graph_file, "rb") as f:
            G = pickle.load(f)  # nosec

        # Load data from input graph
        # TODO we should go through an replace lots of math using self.g_data.* with function IN buckyGraphData
        # TODO toggle spline smoothing
        g_data = buckyGraphData(G, self.sparse, self.consts.diag_Aij)

        # Make contact mats sym and normalized
        self.contact_mats = G.graph["contact_mats"]
        if self.debug:
            logging.debug(f"graph contact mats: {G.graph['contact_mats'].keys()}")
        for mat in self.contact_mats:
            c_mat = xp.array(self.contact_mats[mat])
            c_mat = (c_mat + c_mat.T) / 2.0
            self.contact_mats[mat] = c_mat
        # remove all_locations so we can sum over the them ourselves
        if "all_locations" in self.contact_mats:
            del self.contact_mats["all_locations"]

        # Remove unknown contact mats
        valid_contact_mats = ["home", "work", "other_locations", "school"]
        self.contact_mats = {k: v for k, v in self.contact_mats.items() if k in valid_contact_mats}

        self.Cij = xp.vstack([self.contact_mats[k][None, ...] for k in sorted(self.contact_mats)])

        # Get stratified population (and total)
        self.Nij = g_data.Nij
        self.Nj = g_data.Nj
        self.init_date = g_data.start_date

        self.base_mc_instance = buckyMCInstance(self.init_date, self.t_max, self.Nij, self.Cij)

        # fill in npi_params either from file or as ones
        self.npi_params = get_npi_params(g_data, self.init_date, self.t_max, self.npi_file, self.disable_npi)

        if self.npi_params["npi_active"]:
            self.base_mc_instance.add_npi(self.npi_params)

        if self.consts.vacc_active:
            if SCENARIO_HUB:
                self.vacc_data = buckyVaccAlloc(g_data, self.init_date, self.t_max, self.consts, scen_params)
            else:
                self.vacc_data = buckyVaccAlloc(g_data, self.init_date, self.t_max, self.consts)
            self.base_mc_instance.add_vacc(self.vacc_data)
        return g_data

    # TODO static?
    def calc_lagged_rate(self, var1, var2, lag, mean_days, rollup_func=None):
        """WIP"""

        var1_lagged = frac_last_n_vals(var1, mean_days, axis=0, offset=lag)
        var1_lagged = var1_lagged - frac_last_n_vals(var1, mean_days, axis=0, offset=lag + mean_days + 1)
        var1_var2_ratio = var2 / var1_lagged
        ret = xp.mean(var1_var2_ratio, axis=0)

        # harmonic mean:
        # ret = 1.0 / xp.nanmean(1.0 / var1_var2_ratio, axis=0)

        return ret

    def reset(self, seed=None, params=None):
        """Reset the state of the model and generate new inital data from a new random seed"""
        # TODO we should refactor reset of the compartments to be real pop numbers then /Nij at the end

        # Set random seeds
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(seed)
            xp.random.seed(seed)

        # reroll model params
        # self.g_data.Aij.perturb(self.consts.reroll_variance)
        self.params = self.bucky_params.generate_params()

        # Take a deep copy of the params and ensure they are all correctly on cpu/gpu
        if params is not None:
            self.params = copy.deepcopy(params)

        for k in self.params:
            if type(self.params[k]).__module__ == np.__name__:
                self.params[k] = xp.asarray(self.params[k])

        if self.debug:
            logging.debug("params: " + pformat(self.params, width=120))

        # Reroll vaccine allocation
        if self.base_mc_instance.vacc_data.reroll:
            self.base_mc_instance.vacc_data.reroll_distribution(self.params)
            self.base_mc_instance.vacc_data.reroll_doses(self.params)
            if SCENARIO_HUB:
                self.params["vacc_eff_1"] = scen_params["eff_1"]
                self.params["vacc_eff_2"] = scen_params["eff_2"]

        # TODO move most of below into a function like:
        # test = calc_initial_state(self.g_data, self.params, self.base_mc_instance)

        # Estimate the current age distribution of S, S_age_dist
        if self.base_mc_instance.vacc_data is not None:
            nonvaccs = xp.clip(1 - self.base_mc_instance.vacc_data.V_tot(self.params, 0), a_min=0, a_max=1)  # dose2[0]
        else:
            nonvaccs = 1.0
        tmp = nonvaccs * self.g_data.Nij / self.g_data.Nj
        S_age_dist = tmp / xp.sum(tmp, axis=0)

        # estimate IFR for our age bins
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7721859/
        mean_ages = xp.mean(self.params.consts.age_bins, axis=1)
        ifr = xp.exp(-7.56 + 0.121 * mean_ages) / 100.0

        # Estimate the case reporting rate
        # crr_days_needed = max( #TODO this depends on all the Td params, and D_REPORT_TIME...
        case_reporting = estimate_crr(
            self.g_data,
            self.params,
            cfr=ifr[..., None],  # self.params.F,
            # case_lag=14,
            days_back=25,
            min_deaths=self.consts.case_reporting_min_deaths,
            S_dist=nonvaccs,  # S_age_dist * 16.0,
        )

        self.case_reporting = approx_mPERT(  # TODO these facs should go in param file
            mu=xp.clip(case_reporting, a_min=0.05, a_max=0.95),
            a=xp.clip(0.7 * case_reporting, a_min=0.01, a_max=0.9),
            b=xp.clip(1.3 * case_reporting, a_min=0.1, a_max=1.0),
            gamma=50.0,
        )

        mean_case_reporting = xp.nanmean(self.case_reporting[-self.consts.case_reporting_N_historical_days :], axis=0)

        # Fill in and correct the shapes of some parameters
        self.params["CFR"] = ifr[..., None] * mean_case_reporting[None, ...]
        self.params["CHR"] = xp.broadcast_to(self.params.CHR[:, None], self.Nij.shape)
        self.params["HFR"] = xp.clip(self.params["CFR"] / self.params["CHR"], 0.0, 1.0)
        self.params["CRR"] = mean_case_reporting
        self.params["THETA"] = xp.broadcast_to(
            self.params["THETA"][:, None],
            self.Nij.shape,
        )  # TODO move all the broadcast_to's to one place, they're all over reset()
        self.params["GAMMA_H"] = xp.broadcast_to(self.params["GAMMA_H"][:, None], self.Nij.shape)

        self.params["overall_adm2_ifr"] = xp.sum(ifr[:, None] * self.g_data.Nij / self.g_data.Nj, axis=0)
        # Build init state vector (self.y)
        yy = buckyState(self.consts, self.Nij)

        # Ti = self.params.Ti
        current_I = xp.sum(frac_last_n_vals(self.g_data.rolling_inc_cases, self.params["Ti"], axis=0), axis=0)

        current_I[xp.isnan(current_I)] = 0.0
        current_I[current_I < 0.0] = 0.0
        current_I *= 1.0 / (self.params["CRR"])

        # Roll some random factors for the init compartment values
        # TODO move these inline
        R_fac = self.params.R_fac
        E_fac = self.params.E_fac
        H_fac = self.params.H_fac
        # TODO add an mPERT F_fac instead of the truncnorm

        age_dist_fac = self.g_data.Nij / self.g_data.Nj[None, ...]
        I_init = E_fac * current_I[None, :] * S_age_dist / self.Nij
        D_init = self.g_data.cum_death_hist[-1][None, :] * age_dist_fac / self.Nij
        recovered_init = (self.g_data.cum_case_hist[-1] / self.params["SYM_FRAC"]) * R_fac
        R_init = (
            (recovered_init) * age_dist_fac / self.Nij - D_init - I_init / self.params["SYM_FRAC"]
        )  # Rh is factored in later

        Rt = estimate_Rt(self.g_data, self.params, 14, self.case_reporting)
        Rt = Rt * self.params.Rt_fac

        self.params["R0"] = Rt
        self.params["BETA"] = Rt * self.params["GAMMA"] / self.g_data.Aij.diag

        exp_frac = (
            E_fac
            * xp.ones(I_init.shape[-1])
            * (self.params.R0)
            * self.params.GAMMA
            / self.params.SIGMA
            / (1.0 - R_init)
            / self.params["SYM_FRAC"]
        )

        self.params["CHR"] = estimate_chr(self.g_data, self.params, S_age_dist, days_back=7)
        yy.I = (1.0 - self.params.CHR * self.params["CRR"]) * I_init / yy.Im
        yy.Ic = self.params.CHR * I_init / yy.Im * self.params["CRR"]
        # rh_fac = self.consts.rh_scaling
        yy.Rh = self.params.CHR * I_init / yy.Rhn * self.params["CRR"]

        if self.consts.rescale_chr:
            adm1_hosp = xp.zeros((self.g_data.max_adm1 + 1,), dtype=float)
            xp.scatter_add(adm1_hosp, self.g_data.adm1_id, xp.sum(yy.Rh * self.Nij, axis=(0, 1)))
            adm2_hosp_frac = (self.g_data.adm1_curr_hosp_hist[-1] / adm1_hosp)[self.g_data.adm1_id]
            adm0_hosp_frac = xp.nansum(self.g_data.adm1_curr_hosp_hist[-1]) / xp.nansum(adm1_hosp)
            adm2_hosp_frac[xp.isnan(adm2_hosp_frac) | (adm2_hosp_frac == 0.0)] = adm0_hosp_frac

            # adm2_hosp_frac = xp.sqrt(adm2_hosp_frac * adm0_hosp_frac)

            scaling_H = adm2_hosp_frac * H_fac  # * self.consts.F_scaling
            F_RR_fac = xp.broadcast_to(self.params.F_fac / H_fac, (adm1_hosp.size,)) * H_fac  # /scaling_H
            self.params["CFR"] = estimate_cfr(self.g_data, self.params, S_age_dist, days_back=7)
            self.params["CFR"] = xp.clip(
                self.params["CFR"] * self.consts.F_scaling * F_RR_fac[self.g_data.adm1_id] * scaling_H, 0.0, 1.0
            )
            self.params["CHR"] = xp.clip(self.params["CHR"] * scaling_H, self.params["CFR"], 1.0)

            adm2_chr_delay = xp.sum(
                self.params["I_TO_H_TIME"][:, None] * S_age_dist,
                axis=0,
            )
            adm2_chr_delay_int = adm2_chr_delay.astype(int)  # TODO temp, this should be a distribution of floats
            adm2_chr_delay_mod = adm2_chr_delay % 1
            inc_case_h_delay = (1.0 - adm2_chr_delay_mod) * xp.take_along_axis(
                self.g_data.rolling_inc_cases,
                -adm2_chr_delay_int[None, :],
                axis=0,
            )[0] + adm2_chr_delay_mod * xp.take_along_axis(
                self.g_data.rolling_inc_cases,
                -adm2_chr_delay_int[None, :] - 1,
                axis=0,
            )[
                0
            ]
            inc_case_h_delay[(inc_case_h_delay > 0.0) & (inc_case_h_delay < 1.0)] = 1.0
            inc_case_h_delay[inc_case_h_delay < 0.0] = 0.0
            adm2_chr = xp.sum(self.params["CHR"] * S_age_dist, axis=0)

            tmp = (
                xp.sum(self.params.CHR * I_init / yy.Im * self.g_data.Nij, axis=0) / self.params["CRR"]
            ) * self.params.GAMMA_H  # * self.params.SIGMA #** 2
            tmp2 = inc_case_h_delay * adm2_chr

            ic_fac = tmp2 / tmp
            ic_fac[~xp.isfinite(ic_fac)] = xp.nanmean(ic_fac[xp.isfinite(ic_fac)])
            # ic_fac = xp.clip(ic_fac, a_min=0.2, a_max=5.0)  #####

            self.params["HFR"] = xp.clip(self.params["CFR"] / self.params["CHR"], 0.0, 1.0)
            yy.I = (1.0 - self.params.CHR * self.params["CRR"]) * I_init / yy.Im  # * 0.8
            yy.Ic *= ic_fac * 0.75  # * 0.9 * .9
            yy.Rh *= 1.0 * adm2_hosp_frac

        R_init -= xp.sum(yy.Rh, axis=0)

        yy.Ia = self.params.ASYM_FRAC / self.params.SYM_FRAC * I_init / yy.Im
        yy.E = exp_frac[None, :] * I_init / yy.En  # this should be calcable from Rt and the time before symp
        yy.R = xp.clip(R_init, a_min=0.0, a_max=None)
        yy.D = D_init

        # TMP
        yy.state = xp.clip(yy.state, a_min=0.0, a_max=None)
        mask = xp.sum(yy.N, axis=0) > 1.0
        yy.state[:, mask] /= xp.sum(yy.N, axis=0)[mask]
        mask = xp.sum(yy.N, axis=0) < 1.0
        yy.S[mask] /= 1.0 - xp.sum(yy.N, axis=0)[mask]

        yy.init_S()
        # init the bin we're using to track incident cases
        # (it's filled with cumulatives until we diff it later)
        # TODO should this come from the rolling hist?
        yy.incC = xp.clip(self.g_data.cum_case_hist[-1][None, :], a_min=0.0, a_max=None) * S_age_dist / self.Nij

        self.y = yy

        # Sanity check state vector
        self.y.validate_state()
        # Reroll vaccine calculations if were running those
        if self.base_mc_instance.vacc_data is not None and self.base_mc_instance.vacc_data.reroll:
            self.base_mc_instance.vacc_data.reroll_distribution(self.params)
            self.base_mc_instance.vacc_data.reroll_doses(self.params)
            # if SCENARIO_HUB:
            #     self.params["vacc_eff_1"] = scen_params["eff_1"]
            #     self.params["vacc_eff_2"] = scen_params["eff_2"]

        if self.debug:
            logging.debug("done model reset with seed " + str(seed))

        # return y

    def run_once(self, seed=None):
        """Perform one complete run of the simulation"""
        # rename to integrate or something? it also resets...

        # reset everything
        logging.debug("Resetting state")
        self.reset(seed=seed)
        logging.debug("Done reset")

        self.base_mc_instance.epi_params = self.params
        self.base_mc_instance.state = self.y
        self.base_mc_instance.Aij = self.g_data.Aij.A
        self.base_mc_instance.rhs = RHS_func
        self.base_mc_instance.dy = self.y.zeros_like()

        # TODO this logic needs to go somewhere else (its rescaling beta to account for S/N term)
        # TODO R0 need to be changed before reset()...
        S_eff = self.base_mc_instance.S_eff(0, self.base_mc_instance.state)
        adm2_S_eff = xp.sum(S_eff * self.g_data.Nij / self.g_data.Nj, axis=0)
        adm2_beta_scale = xp.clip(1.0 / (adm2_S_eff + 1e-10), a_min=0.1, a_max=10.0)
        adm1_S_eff = xp.sum(self.g_data.sum_adm1((S_eff * self.g_data.Nij).T).T / self.g_data.adm1_Nj, axis=0)
        adm1_beta_scale = xp.clip(1.0 / (adm1_S_eff + 1e-10), a_min=0.1, a_max=10.0)
        adm2_beta_scale = adm1_beta_scale[self.g_data.adm1_id]

        # adm2_beta_scale = xp.sqrt(adm2_beta_scale)

        self.base_mc_instance.epi_params["R0"] = self.base_mc_instance.epi_params["R0"] * adm2_beta_scale
        self.base_mc_instance.epi_params["BETA"] = self.base_mc_instance.epi_params["BETA"] * adm2_beta_scale
        adm2_E_tot = xp.sum(self.y.E * self.g_data.Nij / self.g_data.Nj, axis=(0, 1))
        adm2_new_E_tot = adm2_beta_scale * adm2_E_tot
        S_dist = S_eff / (xp.sum(S_eff, axis=0) + 1e-10)

        new_E = xp.tile(
            (S_dist * adm2_new_E_tot / self.g_data.Nij * self.g_data.Nj / self.params.consts["En"])[None, ...],
            (xp.to_cpu(self.params.consts["En"]), 1, 1),
        )
        new_S = self.y.S - xp.sum(new_E - self.y.E, axis=0)

        self.base_mc_instance.state.E = new_E
        self.base_mc_instance.state.S = new_S

        self.base_mc_instance.epi_params["BETA"] = xp.broadcast_to(
            self.base_mc_instance.epi_params["BETA"],
            self.g_data.Nij.shape,
        )
        # do integration
        logging.debug("Starting integration")
        sol = xp_ivp.solve_ivp(**self.base_mc_instance.integrator_args)
        logging.debug("Done integration")

        return sol

    def run_multiple(self, n_mc, base_seed=42, out_columns=None, invalid_ret=None):
        """Perform multiple monte carlos and return their postprocessed results"""
        seed_seq = np.random.SeedSequence(base_seed)
        success = 0
        fail = 0
        ret = []
        pbar = tqdm.tqdm(total=n_mc, desc="Performing Monte Carlos", dynamic_ncols=True)
        while success < n_mc:
            mc_seed = seed_seq.spawn(1)[0].generate_state(1)[0]  # inc spawn key then grab next seed
            pbar.set_postfix_str(
                "seed=" + str(mc_seed),
                refresh=True,
            )
            try:
                if fail > n_mc:
                    return invalid_ret

                with xp.optimize_kernels():
                    sol = self.run_once(seed=mc_seed)
                    mc_data = self.postprocess_run(sol, mc_seed, out_columns)
                ret.append(mc_data)
                success += 1
                pbar.update(1)
            except SimulationException:
                fail += 1

            except ValueError:
                fail += 1
                print("nan in rhs")

        pbar.close()
        return ret

    # TODO Also provide methods like to_dlpack, to_pytorch, etc
    def save_run(self, sol, seed):
        """Postprocess and write to disk the output of run_once"""

        mc_data = self.postprocess_run(sol, seed)

        self.writer.write_mc_data(mc_data)

        # TODO write params out (to yaml?) in another subfolder

        # TODO we should output the per monte carlo param rolls, this got lost when we switched from hdf5

    def postprocess_run(self, sol, seed, columns=None):
        """Process the output of a run (sol, returned by the integrator) into the requested output vars"""
        if columns is None:
            columns = [
                "adm2_id",
                "date",
                "rid",
                "total_population",
                "current_hospitalizations",
                "active_asymptomatic_cases",
                "cumulative_deaths",
                "daily_hospitalizations",
                "daily_cases",
                "daily_reported_cases",
                "daily_deaths",
                "cumulative_cases",
                "cumulative_reported_cases",
                "current_icu_usage",
                "current_vent_usage",
                "case_reporting_rate",
                "R_eff",
            ]

            columns = set(columns)

        mc_data = {}

        out = buckyState(self.consts, self.Nij)

        y = sol.y.reshape(self.y.state_shape + (sol.y.shape[-1],))

        # rescale by population
        out.state = self.Nij[None, ..., None] * y

        # collapse age groups
        out.state = xp.sum(out.state, axis=1)

        # population_conserved = (xp.diff(xp.around(xp.sum(out.N, axis=(0, 1)), 1)) == 0.0).all()
        # if not population_conserved:
        #    pass  # TODO we're getting small fp errors here
        #    # print(xp.sum(xp.diff(xp.around(xp.sum(out[:incH], axis=(0, 1)), 1))))
        #    # logging.error("Population not conserved!")
        #    # print(xp.sum(xp.sum(y[:incH],axis=0)-1.))
        #    # raise SimulationException

        if "adm2_id" in columns:
            adm2_ids = np.broadcast_to(self.g_data.adm2_id[:, None], out.state.shape[1:])
            mc_data["adm2_id"] = adm2_ids

        if "date" in columns:
            if self.output_dates is None:
                t_output = xp.to_cpu(sol.t)
                dates = [str(self.init_date + datetime.timedelta(days=np.round(t))) for t in t_output]
                self.output_dates = dates

            mc_data["date"] = np.broadcast_to(np.arange(len(self.output_dates)), out.state.shape[1:])

        if "rid" in columns:
            mc_data["rid"] = np.broadcast_to(seed, out.state.shape[1:])

        if "current_icu_usage" in columns or "current_vent_usage" in columns:
            icu = self.Nij[..., None] * self.params["ICU_FRAC"][:, None, None] * xp.sum(y[out.indices["Rh"]], axis=0)
            if "current_icu_usage" in columns:
                mc_data["current_icu_usage"] = xp.sum(icu, axis=0)

            if "current_vent_usage" in columns:
                vent = self.params.ICU_VENT_FRAC[:, None, None] * icu
                mc_data["current_vent_usage"] = xp.sum(vent, axis=0)

        if "daily_deaths" in columns:
            daily_deaths = xp.gradient(out.D, axis=-1, edge_order=2)
            mc_data["daily_deaths"] = daily_deaths

            if self.reject_runs:
                init_inc_death_mean = xp.mean(xp.sum(daily_deaths[:, 1:4], axis=0))
                hist_inc_death_mean = xp.mean(xp.sum(self.g_data.inc_death_hist[-7:], axis=-1))

                inc_death_rejection_fac = 2.0  # TODO These should come from the cli arg -r
                if (init_inc_death_mean > inc_death_rejection_fac * hist_inc_death_mean) or (
                    inc_death_rejection_fac * init_inc_death_mean < hist_inc_death_mean
                ):
                    logging.info("Inconsistent inc deaths, rejecting run")
                    raise SimulationException

        if "daily_cases" in columns or "daily_reported_cases" in columns:
            daily_reported_cases = xp.gradient(out.incC, axis=-1, edge_order=2)

            if self.reject_runs:
                init_inc_case_mean = xp.mean(xp.sum(daily_reported_cases[:, 1:4], axis=0))
                hist_inc_case_mean = xp.mean(xp.sum(self.g_data.inc_case_hist[-7:], axis=-1))

                inc_case_rejection_fac = 1.5  # TODO These should come from the cli arg -r
                if (init_inc_case_mean > inc_case_rejection_fac * hist_inc_case_mean) or (
                    inc_case_rejection_fac * init_inc_case_mean < hist_inc_case_mean
                ):
                    logging.info("Inconsistent inc cases, rejecting run")
                    raise SimulationException

            if "daily_reported_cases" in columns:
                mc_data["daily_reported_cases"] = daily_reported_cases

            if "daily_cases" in columns:
                daily_cases_total = daily_reported_cases / self.params.CRR[:, None]
                mc_data["daily_cases"] = daily_cases_total

        if "cumulative_reported_cases" in columns:
            cum_cases_reported = out.incC
            mc_data["cumulative_reported_cases"] = cum_cases_reported

        if "cumulative_cases" in columns:
            cum_cases_total = out.incC / self.params.CRR[:, None]
            mc_data["cumulative_cases"] = cum_cases_total

        if "daily_hospitalizations" in columns:
            out.incH[:, 0] = out.incH[:, 1]
            daily_hosp = xp.gradient(out.incH, axis=-1, edge_order=2)
            mc_data["daily_hospitalizations"] = daily_hosp

        if "total_population" in columns:
            N = xp.broadcast_to(self.g_data.Nj[..., None], out.state.shape[1:])
            mc_data["total_population"] = N

        if "current_hospitalizations" in columns:
            hosps = xp.sum(out.Rh, axis=0)  # why not just using .H?
            mc_data["current_hospitalizations"] = hosps

        if "cumulative_deaths" in columns:
            cum_deaths = out.D
            mc_data["cumulative_deaths"] = cum_deaths

        if "active_asymptomatic_cases" in columns:
            asym_I = xp.sum(out.Ia, axis=0)
            mc_data["active_asymptomatic_cases"] = asym_I

        if "case_reporting_rate" in columns:
            crr = xp.broadcast_to(self.params.CRR[:, None], adm2_ids.shape)
            mc_data["case_reporting_rate"] = crr

        if "R_eff" in columns:
            #    r_eff = self.npi_params["r0_reduct"].T * np.broadcast_to(
            #        (self.params.R0 * self.g_data.Aij.diag)[:, None], adm2_ids.shape
            #    )
            mc_data["R_eff"] = xp.broadcast_to(self.params.R0[:, None], adm2_ids.shape)

        if self.consts.vacc_reroll:
            dose1 = xp.sum(self.base_mc_instance.vacc_data.dose1 * self.Nij[None, ...], axis=1).T
            dose2 = xp.sum(self.base_mc_instance.vacc_data.dose2 * self.Nij[None, ...], axis=1).T
            mc_data["vacc_dose1"] = dose1
            mc_data["vacc_dose2"] = dose2
            dose1_65 = xp.sum((self.base_mc_instance.vacc_data.dose1 * self.Nij[None, ...])[:, -3:], axis=1).T
            dose2_65 = xp.sum((self.base_mc_instance.vacc_data.dose2 * self.Nij[None, ...])[:, -3:], axis=1).T
            mc_data["vacc_dose1_65"] = dose1_65
            mc_data["vacc_dose2_65"] = dose2_65

            pop = xp.sum((self.Nij[None, ...]), axis=1).T
            pop_65 = xp.sum((self.Nij[None, ...])[:, -3:], axis=1).T

            mc_data["frac_vacc_dose1"] = dose1 / pop
            mc_data["frac_vacc_dose2"] = dose2 / pop
            mc_data["frac_vacc_dose1_65"] = dose1_65 / pop_65
            mc_data["frac_vacc_dose2_65"] = dose2_65 / pop_65

            tmp = buckyState(self.consts, self.Nij)
            v_eff = xp.zeros_like(self.base_mc_instance.vacc_data.dose1)
            for i in range(y.shape[-1]):
                tmp.state = y[..., i]
                v_eff[i] = self.base_mc_instance.vacc_data.V_eff(tmp, self.params, i) + tmp.R

            imm = xp.sum(v_eff * self.Nij[None, ...], axis=1).T
            imm_65 = xp.sum((v_eff * self.Nij[None, ...])[:, -3:], axis=1).T
            mc_data["immune"] = imm
            mc_data["immune_65"] = imm_65
            mc_data["frac_immune"] = imm / pop
            mc_data["frac_immune_65"] = imm_65 / pop_65

            # phase
            mc_data["state_phase"] = self.base_mc_instance.vacc_data.phase_hist.T

        # Collapse the gamma-distributed compartments and move everything to cpu
        negative_values = False
        for k, val in mc_data.items():
            if k != "date" and xp.any(xp.around(val, 2) < 0.0):
                logging.info("Negative values present in " + k)
                negative_values = True

        if negative_values and self.reject_runs:
            logging.info("Rejecting run b/c of negative values in output")
            raise SimulationException

        self.writer.write_params(seed, self.params)

        return mc_data


def main(args=None):
    """Main method for a complete simulation called with a set of CLI args"""
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args=args)

    if args.gpu:
        logging.info("Using GPU backend")
        enable_cupy(optimize=args.optimize_kernels)

    reimport_numerical_libs("model.main.main")

    warnings.simplefilter(action="ignore", category=xp.ExperimentalWarning)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    loglevel = 30 - 10 * min(args.verbosity, 2)

    logging.getLogger().setLevel(loglevel)
    debug_mode = loglevel < 20

    # TODO we should output the logs to output_dir too...
    _banner()

    with logging_redirect_tqdm():
        logging.info(f"command line args: {args}")
        env = buckyModelCovid(
            debug=debug_mode,
            sparse_aij=(not args.dense),
            t_max=args.days,
            graph_file=args.graph_file,
            par_file=args.par_file,
            npi_file=args.npi_file,
            disable_npi=args.disable_npi,
            reject_runs=args.reject_runs,
            output_dir=args.output_dir,
        )

        try:
            if args.optimize:
                test_opt(env)
                return
                # TODO Should exit() here

            seed_seq = np.random.SeedSequence(args.seed)

            pbar = tqdm.tqdm(total=args.n_mc, desc="Performing Monte Carlos", dynamic_ncols=True)
            total_start = datetime.datetime.now()
            success = 0
            n_runs = 0

            while success < args.n_mc:
                mc_seed = seed_seq.spawn(1)[0].generate_state(1)[0]  # inc spawn key then grab next seed
                pbar.set_postfix_str(
                    "seed=" + str(mc_seed),
                    # + ", rej%="  # TODO disable rej% if not -r
                    # + str(np.around(float(n_runs - success) / (n_runs + 0.00001) * 100, 1)),
                    # refresh=True,
                )
                try:
                    n_runs += 1
                    with xp.optimize_kernels():
                        sol = env.run_once(seed=mc_seed)
                        env.save_run(sol, mc_seed)

                    success += 1
                    pbar.update(1)
                except SimulationException as e:
                    # print(e)
                    pass

        except (KeyboardInterrupt, SystemExit):
            logging.warning("Caught SIGINT, cleaning up")
            env.writer.close()  # TODO need env.close() which checks if writer is inited
        finally:
            env.writer.close()
            if "pbar" in locals():
                pbar.close()
                logging.info(f"Total runtime: {datetime.datetime.now() - total_start}")


if __name__ == "__main__":
    main()
