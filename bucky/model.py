import copy
import csv
import datetime
import glob
import logging
import os
import pickle
import random
import sys
from collections import defaultdict, deque
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .arg_parser_model import parser
from .parameters import seir_params
from .util import (
    _banner,
    cache_files,
    dotdict,
    import_numerical_libs,
    map_np_array,
    truncnorm,
)

if __name__ == "__main__":
    args = parser.parse_args()
    import_numerical_libs(args.gpu)
else:  # skip arg parse for sphinx
    import_numerical_libs(False)

from .util.util import ivp, sparse, xp  # isort:skip

#
# Params TODO move all this to arg_parser or elsewhere
#
OUTPUT = True

# TODO move to param file
RR_VAR = 0.3  # variance to use for MC of params with no CI


class SimulationException(Exception):
    pass


class SEIR_covid(object):
    def __init__(self, seed=None, randomize_params_on_reset=True):
        self.rseed = seed
        self.randomize = randomize_params_on_reset

        # Integrator params
        self.t = 0.0
        self.dt = 1.0  # time step for model output (the internal step is adaptive...)
        self.t_max = args.days
        self.done = False
        start = datetime.datetime.now()
        self.mc_id = str(start).replace(" ", "__").replace(":", "_").split(".")[0]
        logging.info(f"MC ID: {self.mc_id}")

        self.G = None
        self.graph_file = args.graph_file

        # save files to cache
        if args.cache:
            logger.warn(
                "Cacheing is currently unsupported and probably doesnt work after the refactor"
            )
            files = glob.glob("*.py") + [self.graph_file, args.par_file]
            logging.info(f"Cacheing: {files}")
            cache_files(files, self.mc_id)

        # disease params
        self.s_par = seir_params(args.par_file, args.gpu)
        self.model_struct = self.s_par.generate_params(None)["model_struct"]

        global Si, Ei, Ii, Ici, Iasi, Ri, Rhi, Di, Iai, Hi, Ci, N_compartments, En, Im, Rhn
        En = self.model_struct["En"]
        Im = self.model_struct["Im"]
        Rhn = self.model_struct["Rhn"]
        Si = 0
        Ei = xp.array(Si + 1 + xp.arange(En), dtype=int)
        Ii = xp.array(Ei[-1] + 1 + xp.arange(Im), dtype=int)
        Ici = xp.array(Ii[-1] + 1 + xp.arange(Im), dtype=int)
        Iasi = xp.array(Ici[-1] + 1 + xp.arange(Im), dtype=int)
        Ri = Iasi[-1] + 1
        Rhi = xp.array(Ri + 1 + xp.arange(Rhn), dtype=int)
        Di = Rhi[-1] + 1

        Iai = xp.hstack([Ii, Iasi, Ici])  # all I compartmetns
        Hi = xp.hstack([Rhi, Ici])  # all compartments in hospitalization
        Ci = xp.hstack([Ii, Ici, Rhi])

        N_compartments = Di.get() + 1 if "cupy" in type(Di).__module__ else Di + 1

    def reset(self, seed=None, params=None):

        # if you set a seed using the constructor, you're stuck using it forever
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.rseed = seed

        #
        # Init graph
        #

        self.t = 0.0
        self.iter = 0
        self.done = False

        if self.G is None:

            logging.info("loading graph")
            with open(self.graph_file, "rb") as f:
                G = pickle.load(f)

            # get initial case data

            self.init_cum_cases = xp.array(
                list(nx.get_node_attributes(G, "Confirmed").values())
            ).astype(float)
            self.init_deaths = xp.array(
                list(nx.get_node_attributes(G, "Deaths").values())
            ).astype(float)
            self.init_cum_cases[self.init_cum_cases < 0.0] = 0.0

            case_hist = xp.vstack(
                list(nx.get_node_attributes(G, "case_hist").values())
            ).T

            self.case_hist_cum = case_hist.astype(float)
            self.case_hist = xp.diff(case_hist, axis=0).astype(
                float
            )  # TODO rename to daily_case_hist

            death_hist = xp.vstack(
                list(nx.get_node_attributes(G, "death_hist").values())
            ).T

            self.death_hist_cum = death_hist.astype(float)
            self.death_hist = xp.diff(death_hist, axis=0).astype(float)  # TODO rename

            # grab the geo id's for later
            self.adm2_id = np.fromiter(
                nx.get_node_attributes(G, G.graph["adm2_key"]).values(), dtype=int
            )

            # dict keyed by ADM1 that has a list of indices in that ADM1
            self.state_map = defaultdict(partial(np.empty, (0,), int))
            for k, v in nx.get_node_attributes(G, G.graph["adm1_key"]).items():
                self.state_map[v] = np.append(self.state_map[v], k)
            self.adm1_id = np.fromiter(
                nx.get_node_attributes(G, G.graph["adm1_key"]).values(), dtype=int
            )

            # Make contact mats sym and normalized
            self.contact_mats = G.graph["contact_mats"]
            logging.debug(f"graph contact mats: {G.graph['contact_mats']}")
            self.contact_mats["NPI"] = (
                self.contact_mats["home"]
                + self.contact_mats["other_locations"]
                + 0.5 * self.contact_mats["work"]
            )
            for mat in self.contact_mats:
                c_mat = self.contact_mats[mat]
                c_mat = (c_mat + c_mat.T) / 2.0
                c_mat /= np.sum(c_mat, axis=0)[None, :]
                self.contact_mats[mat] = c_mat

            self.contact_mat = xp.array(self.contact_mats["NPI"])  # ["all_locations"]
            # Get stratified population (and total)
            self.n_age_grps = self.contact_mat.shape[0]  # s["all_locations"].shape[0]
            N_age_init = nx.get_node_attributes(G, "N_age_init")
            self.Nij = xp.asarray((np.vstack(list(N_age_init.values())) + 0.0001).T)
            self.Nj = xp.asarray(np.sum(self.Nij, axis=0))

            logging.info("done")

            self.G = G

            self.first_date = datetime.date.fromisoformat(G.graph["start_date"])
            # Build adj mat for the RHS
            G = nx.convert_node_labels_to_integers(G)

            edges = xp.array(list(G.edges(data="weight"))).T

            A = sparse.coo_matrix(
                (edges[2], (edges[0].astype(int), edges[1].astype(int)))
            )
            A = A.toarray()

            self.baseline_A = A / xp.sum(A, axis=0)

        # make sure we always reset to baseline
        self.A = self.baseline_A

        # randomize model params if we're doing that kind of thing
        if self.randomize:
            self.reset_A(RR_VAR)
            self.params = self.s_par.generate_params(RR_VAR)

        else:
            self.params = self.s_par.generate_params(None)

        if params is not None:
            self.params = copy.deepcopy(params)

        logging.info(f"params: {self.params}")

        for k in self.params:
            if type(self.params[k]).__module__ == np.__name__:
                self.params[k] = xp.asarray(self.params[k])

        self.case_reporting = self.estimate_reporting(days_back=22)

        self.doubling_t = self.estimate_doubling_time()

        if xp.any(~xp.isfinite(self.doubling_t)):
            raise RuntimeError("non finite doubling times, is there enough case data?")

        if RR_VAR > 0.0:
            self.doubling_t *= truncnorm(
                xp, 1.0, RR_VAR, size=self.doubling_t.shape, a_min=1e-6
            )
            self.doubling_t = xp.clip(self.doubling_t, 1.0, None)

        self.params = self.s_par.rescale_doubling_rate(
            self.doubling_t, self.params, xp, self.A
        )

        n_nodes = len(self.G.nodes())
        # hist_weights = xp.arange(1.0, self.case_reporting.shape[0] + 1.0, 1.0)
        # hist_case_reporting = xp.sum(
        #    self.case_reporting * hist_weights[:, None], axis=0
        # ) / xp.sum(hist_weights)

        hist_case_reporting = xp.mean(self.case_reporting[:-4], axis=0)

        self.params["CASE_REPORT"] = hist_case_reporting
        self.params["F"] = xp.tile(self.params.F[:, None], n_nodes)
        self.params["H"] = (
            xp.tile(self.params.H[:, None], n_nodes)
            * 8398.0
            / 11469.0
            * 8353.0
            / 17530.0
        )
        self.params["THETA"] = xp.broadcast_to(
            self.params["THETA"][:, None], (self.n_age_grps, n_nodes)
        )
        self.params["GAMMA_H"] = xp.broadcast_to(
            self.params["GAMMA_H"][:, None], (self.n_age_grps, n_nodes)
        )
        self.params["F_eff"] = self.params["F"] / self.params["H"]

        # init state vector (self.y)
        y = xp.zeros((N_compartments, self.n_age_grps, n_nodes))

        # Init S=1 everywhere
        y[Si, :, :] = 1.0

        logging.debug("case init")
        Ti = self.params.Ti
        current_I = (
            xp.sum(self.case_hist[-int(Ti) :], axis=0)
            + (Ti % 1) * self.case_hist[-int(Ti + 1)]
        )
        current_I[xp.isnan(current_I)] = 0.0
        current_I[current_I < 0.0] = 0.0
        current_I *= 1.0 / (self.params["CASE_REPORT"])

        I_init = current_I[None, :] / self.Nij / self.n_age_grps
        D_init = self.init_deaths[None, :] / self.Nij / self.n_age_grps
        recovered_init = (
            (self.init_cum_cases)
            / self.params["SYM_FRAC"]
            / (self.params["CASE_REPORT"])
        )
        R_init = (
            (recovered_init) / self.Nij / self.n_age_grps
            - D_init
            - I_init / self.params["SYM_FRAC"]
        )  # rhi handled later

        ic_frac = 1.0 / (1.0 + self.params.THETA / self.params.GAMMA_H)
        hosp_frac = 1.0 / (1.0 + self.params.GAMMA_H / self.params.THETA)
        exp_frac = (
            1.0
            * xp.ones(I_init.shape[-1])  # 5
            # * np.diag(self.A)
            # * np.sum(self.A, axis=1)
            * (self.params.R0 @ self.A)
            * self.params.GAMMA
            / self.params.SIGMA
        )

        y[Si] -= I_init
        y[Ii] = (1.0 - self.params.H) * I_init / len(Ii)

        y[Ici] = ic_frac * self.params.H * I_init / (len(Ici))
        y[Rhi] = hosp_frac * self.params.H * I_init / (Rhn)
        # y[Ici] = self.params.H * I_init / (len(Ici)+Rhn)
        # y[Rhi] = (
        #    # (self.params.THETA / self.params.GAMMA_H) *
        #    np.sum(y[Ici], axis=0)
        #    / (len(Ici)+len(Rhi))
        # )
        # y[Si] -= cp.sum(y[Rhi], axis=0)

        R_init -= xp.sum(y[Rhi], axis=0)

        y[Si] -= self.params.ASYM_FRAC * I_init
        y[Iasi] = self.params.ASYM_FRAC * I_init / len(Iasi)
        y[Si] -= exp_frac[None, :] * I_init
        y[Ei] = exp_frac[None, :] * I_init / len(Ei)
        y[Si] -= R_init
        y[Ri] = R_init
        y[Si] -= D_init
        y[Di] = D_init

        self.y = y

        if xp.any(~xp.isfinite(self.y)):
            raise RuntimeError(
                "nonfinite values in the state vector, something is wrong with init"
            )

        logging.debug("done reset")

        return y

    def reset_A(self, var):
        new_R0_fracij = truncnorm(xp, 1.0, var, size=self.A.shape, a_min=1e-6)
        new_R0_fracij = xp.clip(new_R0_fracij, 1e-6, None)
        A = self.baseline_A * new_R0_fracij
        self.A = A / xp.sum(A, axis=0)

    def estimate_doubling_time(self, days_back=14, time_window=7, min_doubling_t=1.0):

        cases = self.case_hist_cum[-days_back:] / self.case_reporting[-days_back:]
        cases_old = (
            self.case_hist_cum[-days_back - time_window : -time_window]
            / self.case_reporting[-days_back - time_window : -time_window]
        )

        # adm0
        adm0_doubling_t = (
            time_window
            * xp.log(2.0)
            / xp.log(xp.sum(cases, axis=1) / xp.sum(cases_old, axis=1))
        )

        doubling_t = xp.repeat(adm0_doubling_t[:, None], cases.shape[-1], axis=1)

        # adm1
        for state_ids in self.state_map.values():
            adm1_doubling_t = (
                time_window
                * xp.log(2.0)
                / xp.log(
                    xp.sum(cases[:, state_ids], axis=1)
                    / xp.sum(cases_old[:, state_ids], axis=1)
                )
            )

            valid_values = xp.isfinite(adm1_doubling_t) & (
                adm1_doubling_t > min_doubling_t
            )
            # ValueError: currently, CuPy only supports slices that consist of one boolean array.
            # case_report[valid_values,state_ids] = adm1_case_report[valid_values]
            for i, valid in enumerate(valid_values):
                if valid:
                    doubling_t[i, state_ids] = adm1_doubling_t[i]

        # adm2
        adm2_doubling_t = time_window * xp.log(2.0) / xp.log(cases / cases_old)

        valid_adm2_dt = xp.isfinite(adm2_doubling_t) & (
            adm2_doubling_t > min_doubling_t
        )
        doubling_t[valid_adm2_dt] = adm2_doubling_t[valid_adm2_dt]

        # hist_weights = xp.arange(1., days_back + 1.0, 1.0)
        # hist_doubling_t = xp.sum(doubling_t * hist_weights[:, None], axis=0) / xp.sum(
        #    hist_weights
        # )
        hist_doubling_t = xp.mean(doubling_t, axis=0) / 2.0

        return hist_doubling_t

    # Todo this should get case_lag from params['D_REPORT_TIME']
    def estimate_reporting(self, days_back=14, case_lag=None, min_deaths=100.0):

        if case_lag is None:
            adm0_cfr_by_age = (
                self.params.F * xp.sum(self.Nij, axis=1) / xp.sum(self.Nj, axis=0)
            )
            adm0_cfr_total = xp.sum(
                self.params.F * xp.sum(self.Nij, axis=1) / xp.sum(self.Nj, axis=0),
                axis=0,
            )
            case_lag = xp.sum(
                self.params["D_REPORT_TIME"] * adm0_cfr_by_age / adm0_cfr_total, axis=0
            )

        case_lag_int = int(case_lag)
        case_lag_frac = case_lag % 1
        cases_lagged = (
            self.case_hist_cum[-case_lag_int - days_back : -case_lag_int]
            + case_lag_frac
            * self.case_hist_cum[-case_lag_int - 1 - days_back : -case_lag_int - 1]
        )

        # adm0
        adm0_cfr_param = xp.sum(
            self.params.F * xp.sum(self.Nij, axis=1) / xp.sum(self.Nj, axis=0), axis=0
        )
        adm0_cfr_reported = xp.sum(self.death_hist_cum[-days_back:], axis=1) / xp.sum(
            cases_lagged, axis=1
        )
        adm0_case_report = adm0_cfr_param / adm0_cfr_reported

        case_report = xp.repeat(
            adm0_case_report[:, None], cases_lagged.shape[-1], axis=1
        )

        # adm1
        for state_ids in self.state_map.values():
            state_deaths = xp.sum(self.death_hist_cum[-days_back:, state_ids], axis=1)
            adm1_cfr_param = xp.sum(
                self.params.F
                * xp.sum(self.Nij[:, state_ids], axis=1)
                / xp.sum(self.Nj[state_ids], axis=0),
                axis=0,
            )
            adm1_cfr_reported = xp.sum(
                self.death_hist_cum[-days_back:, state_ids], axis=1
            ) / xp.sum(cases_lagged[:, state_ids], axis=1)
            adm1_case_report = adm1_cfr_param / adm1_cfr_reported

            valid_values = (state_deaths > min_deaths) & xp.isfinite(adm1_case_report)

            # ValueError: currently, CuPy only supports slices that consist of one boolean array.
            # case_report[valid_values,state_ids] = adm1_case_report[valid_values]
            for i, valid in enumerate(valid_values):
                if valid:
                    case_report[i, state_ids] = adm1_case_report[i]

        # adm2
        adm2_cfr_param = xp.sum(self.params.F[:, None] * (self.Nij / self.Nj), axis=0)

        adm2_cfr_reported = self.death_hist_cum[-days_back:] / cases_lagged
        adm2_case_report = adm2_cfr_param / adm2_cfr_reported

        valid_adm2_cr = xp.isfinite(adm2_case_report) & (
            self.death_hist_cum[-days_back:] > min_deaths
        )
        case_report[valid_adm2_cr] = adm2_case_report[valid_adm2_cr]

        return case_report

    #
    # RHS for odes - d(sstate)/dt = F(t, state, *mats, *pars)
    # NB: requires the state vector be 1d
    #

    @staticmethod
    def _dGdt_vec(t, y, Nij, Cij, Aij, par):
        # constraint on values
        lower, upper = (0.0, 1.0)  # bounds for state vars

        # grab index of OOB values so we can zero derivatives (stability...)
        too_low = y <= lower
        too_high = y >= upper

        # reshape to a sane form (compartment, age, node)
        s = y.reshape(N_compartments, Nij.shape[0], -1)

        # y = xp.clip(y, 0., 1.)
        # Clip state to be in bounds (except allocs b/c thats a counter)
        s = xp.maximum(s, xp.ones(s.shape) * lower)
        s = xp.minimum(s, xp.ones(s.shape) * upper)

        # init d(state)/dt
        dG = xp.zeros(s.shape)

        # effective params after damping w/ allocated stuff
        BETA_eff = par["BETA"]
        F_eff = par["F_eff"]
        H = par["H"]
        THETA = par["THETA"]
        GAMMA = par["GAMMA"]
        GAMMA_H = par["GAMMA_H"]
        SIGMA = par["SIGMA"]

        # perturb Aij
        # new_R0_fracij = truncnorm(xp, 1.0, .1, size=Aij.shape, a_min=1e-6)
        # new_R0_fracij = xp.clip(new_R0_fracij, 1e-6, None)
        # A = Aij * new_R0_fracij
        # Aij_eff = A / xp.sum(A, axis=0)
        Aij_eff = Aij

        # Infectivity matrix (I made this name up, idk what its really called)
        I_tot = xp.sum(Nij * s[Iai], axis=0)

        # I_tmp = (Aij.T @ I_tot.T).T
        I_tmp = I_tot @ Aij_eff  # using identity (A@B).T = B.T @ A.T

        beta_mat = s[Si] * (Cij @ I_tmp)
        beta_mat /= Nij

        # dS/dt
        dG[Si] = -BETA_eff * (beta_mat)
        # dE/dt
        dG[Ei[0]] = BETA_eff * (beta_mat) - En * SIGMA * s[Ei[0]]
        dG[Ei[1:]] = En * SIGMA * s[Ei[:-1]] - En * SIGMA * s[Ei[1:]]

        # dI/dt
        dG[Iasi[0]] = (
            par["ASYM_FRAC"] * En * SIGMA * s[Ei[-1]] - Im * GAMMA * s[Iasi[0]]
        )
        dG[Iasi[1:]] = Im * GAMMA * s[Iasi[:-1]] - Im * GAMMA * s[Iasi[1:]]

        dG[Ii[0]] = (
            par["SYM_FRAC"] * (1.0 - H) * En * SIGMA * s[Ei[-1]] - Im * GAMMA * s[Ii[0]]
        )
        dG[Ii[1:]] = Im * GAMMA * s[Ii[:-1]] - Im * GAMMA * s[Ii[1:]]

        # dIc/dt
        dG[Ici[0]] = (
            par["SYM_FRAC"] * H * En * SIGMA * s[Ei[-1]] - Im * GAMMA_H * s[Ici[0]]
        )
        dG[Ici[1:]] = Im * GAMMA_H * s[Ici[:-1]] - Im * GAMMA_H * s[Ici[1:]]

        # dRhi/dt
        dG[Rhi[0]] = (
            Im * GAMMA_H * s[Ici[-1]] - (1.0 - F_eff) * (Rhn * THETA) * s[Rhi[0]]
        )
        dG[Rhi[1:]] = (1.0 - F_eff) * Rhn * THETA * s[Rhi[:-1]] - (1.0 - F_eff) * (
            Rhn * THETA
        ) * s[Rhi[1:]]

        # dR/dt
        dG[Ri] = (
            Im * GAMMA * (s[Ii[-1]] + s[Iasi[-1]])
            + (1.0 - F_eff) * Rhn * THETA * s[Rhi[-1]]
        )

        # dD/dt
        dG[Rhi] -= F_eff * (THETA) * s[Rhi]
        dG[Di] = xp.sum(F_eff * (THETA) * s[Rhi], axis=0)

        # bring back to 1d for the ODE api
        dG = dG.reshape(-1)

        # zero derivatives for things we had to clip if they are going further out of bounds
        # TODO we can probably do this better, we really dont want to branch in the RHS
        if args.gpu:
            low_count = too_low.sum().get().item()
            high_count = too_high.sum().get().item()
        else:
            low_count = too_low.sum()
            high_count = too_high.sum()
        dG[too_low] = xp.maximum(dG[too_low], xp.ones(low_count) * lower)
        dG[too_high] = xp.minimum(dG[too_high], xp.ones(high_count) * upper)

        return dG

    def dGdt_vec(self, t, y, Nij, Cij, Aij, par):
        return self._dGdt_vec(t, y, Nij, Cij, Aij, par)

    def run_once(self, seed=None, outdir="raw_output/", output=True):

        # reset everything
        self.reset(seed=seed)

        # TODO should output the IC here

        # do integration
        t_eval = np.arange(0, self.t_max + self.dt, self.dt)
        sol = ivp.solve_ivp(
            self.dGdt_vec,
            method="RK23",
            t_span=(0.0, self.t_max),
            y0=self.y.reshape(-1),
            t_eval=t_eval,
            args=(self.Nij, self.contact_mat, self.A, self.params),
        )

        y = sol.y.reshape(N_compartments, self.n_age_grps, -1, len(t_eval))

        out = self.Nij[None, ..., None] * y

        # collapse age groups
        out = xp.sum(out, axis=1)

        population_conserved = (
            xp.diff(xp.around(xp.sum(out, axis=(0, 1)), 1)) == 0.0
        ).all()
        if not population_conserved:
            logging.error("Population not conserved!")
            raise SimulationException

        adm2_ids = np.broadcast_to(self.adm2_id[:, None], out.shape[1:])

        n_time_steps = out.shape[-1]

        t_output = sol.t.get() if "cupy" in type(sol.t).__module__ else sol.t
        dates = [
            pd.Timestamp(self.first_date + datetime.timedelta(days=np.round(t)))
            for t in t_output
        ]
        dates = np.broadcast_to(dates, out.shape[1:])

        icu = (
            self.Nij[..., None]
            * self.params["ICU_FRAC"][:, None, None]
            * xp.sum(y[Hi], axis=0)
        )
        vent = self.params.ICU_VENT_FRAC[:, None, None] * icu
        daily_deaths = xp.diff(
            out[Di], prepend=self.death_hist_cum[-1][:, None], axis=-1
        )
        # daily_deaths_reported = daily_deaths * self.params.CASE_REPORT[:,None]
        cum_cases = (
            xp.sum(
                self.Nij[..., None] * (1.0 - xp.sum(y[: Ei[-1] + 1], axis=0)), axis=0
            )
            * self.params.SYM_FRAC
        )
        daily_cases = xp.diff(
            cum_cases,
            prepend=self.case_hist_cum[-1][:, None] / self.params.CASE_REPORT[:, None],
            axis=-1,
        )
        # cum_cases_reported = cum_cases * self.params["SYM_FRAC"] * self.params.CASE_REPORT[:,None]
        daily_cases_reported = (
            daily_cases * self.params.CASE_REPORT[:, None]  # /self.params.SYM_FRAC
        )
        cum_cases_reported = cum_cases * self.params.CASE_REPORT[:, None]

        # if (daily_cases < 0)[..., 1:].any():
        #    logging.error('Negative daily cases')
        #    raise SimulationException

        out = out.reshape(y.shape[0], -1)

        # Grab pretty much everything interesting
        df_data = {
            "ADM2_ID": adm2_ids.reshape(-1),
            "date": dates.reshape(-1),
            "rid": np.full(out.shape[-1], seed).reshape(-1),  # TODO broadcast_to
            "S": out[Si],
            "E": out[Ei],
            "I": out[Ii],
            "Ic": out[Ici],
            "Ia": out[Iasi],
            "R": out[Ri],
            "Rh": out[Rhi],
            "D": out[Di],
            "NC": daily_cases.reshape(-1),
            "NCR": daily_cases_reported.reshape(-1),
            "ND": daily_deaths.reshape(-1),
            "CC": cum_cases.reshape(-1),
            "CCR": cum_cases_reported.reshape(-1),
            "ICU": xp.sum(icu, axis=0).reshape(-1),
            "VENT": xp.sum(vent, axis=0).reshape(-1),
            "CASE_REPORT": np.broadcast_to(
                self.params.CASE_REPORT[:, None], adm2_ids.shape
            ).reshape(-1),
            "Reff": np.broadcast_to(
                (self.params.R0 * (np.diag(self.A)))[:, None], adm2_ids.shape
            ).reshape(-1),
            "doubling_t": np.broadcast_to(
                self.doubling_t[:, None], adm2_ids.shape
            ).reshape(-1),
        }

        # Collapse the gamma-distributed compartments
        for k in df_data:
            if df_data[k].ndim == 2:
                df_data[k] = xp.sum(df_data[k], axis=0)
            if "cupy" in type(df_data[k]).__module__:
                df_data[k] = df_data[k].get()

        # Append data to the hdf5 file
        output_folder = os.path.join(outdir, self.mc_id)
        os.makedirs(output_folder, exist_ok=True)
        out_df = pd.DataFrame(data=df_data)

        # round off any FP error we picked up along the way
        out_df.set_index(["ADM2_ID", "date", "rid"], inplace=True)
        out_df = out_df.apply(np.around, args=(3,))
        out_df.reset_index(inplace=True)

        if output:
            out_df.to_feather(os.path.join(output_folder, str(seed) + ".feather"))

        # TODO we should output the per monte carlo param rolls, this got lost when we switched from hdf5

        # Append params to the hdf5 file TODO  better handle the arrays
        """
        out_params = pd.DataFrame(self.params).assign(rid=seed)
        out_params.to_hdf(
            outdir + "mc--" + self.mc_id + ".h5",
            key="params",
            mode="a",
            append=True,
            format="table",
            data_columns=["rid", "R0"],
        )
        """


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
    )

    _banner()

    if not os.path.exists(args.output_dir):
        logging.info("Creating output directory @ " + args.output_dir)
        os.mkdir(args.output_dir)

    if args.gpu:
        logging.info("Using GPU backend")

    logging.info(f"command line args: {args}")
    if args.no_mc:
        env = SEIR_covid(randomize_params_on_reset=False)
        n_mc = 1
    else:
        env = SEIR_covid(randomize_params_on_reset=True)
        n_mc = args.n_mc

    total_start = datetime.datetime.now()
    seed = 0
    success = 0
    times = []
    pbar = tqdm(total=n_mc)
    while success < n_mc:
        start = datetime.datetime.now()
        try:
            env.run_once(seed=seed, outdir=args.output_dir)
            success += 1
        except SimulationException:
            pass
        seed += 1
        run_time = (datetime.datetime.now() - start).total_seconds()
        times.append(run_time)
        pbar.update(1)

        logging.info(f"{seed}: {datetime.datetime.now() - start}")
    logging.info(f"Total runtime: {datetime.datetime.now() - total_start}")
