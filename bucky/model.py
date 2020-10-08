import copy
import datetime
import glob
import logging
import os
import pickle
import random
import sys
import warnings
from collections import defaultdict, deque
from functools import lru_cache, partial
from pprint import pformat  # TODO set some defaults for width/etc with partial?

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from .arg_parser_model import parser
from .npi import read_npi_file
from .parameters import seir_params
from .util import (
    TqdmLoggingHandler,
    _banner,
    cache_files,
    date_to_t_int,
    dotdict,
    map_np_array,
)
from .util.distributions import mPERT_sample, truncnorm

# supress pandas warning caused by pyarrow
warnings.simplefilter(action="ignore", category=FutureWarning)
# TODO we do alot of allowing div by 0 and then checking for nans later, we should probably refactor that
warnings.simplefilter(action="ignore", category=RuntimeWarning)

from .numerical_libs import use_cupy

if __name__ == "__main__":
    args = parser.parse_args()
    if args.gpu:
        use_cupy(optimize=True)

from .numerical_libs import xp, ivp, sparse # isort:skip

#
# Params TODO move all this to arg_parser or elsewhere
#
OUTPUT = True
REJECT_RUNS = True

# TODO move to param file
RR_VAR = 0.12  # variance to use for MC of params with no CI


class SimulationException(Exception):
    pass


@lru_cache(maxsize=None)
def get_runid(pid=0):
    start = datetime.datetime.now()
    return str(start).replace(" ", "__").replace(":", "_").split(".")[0]


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
        self.run_id = get_runid()
        logging.info(f"Run ID: {self.run_id}")

        self.G = None
        self.graph_file = args.graph_file

        # save files to cache
        if args.cache:
            logging.warn(
                "Cacheing is currently unsupported and probably doesnt work after the refactor"
            )
            files = glob.glob("*.py") + [self.graph_file, args.par_file]
            logging.info(f"Cacheing: {files}")
            cache_files(files, self.run_id)

        # disease params
        self.s_par = seir_params(args.par_file, args.gpu)
        self.model_struct = self.s_par.generate_params(None)["model_struct"]

        #global Si, Ei, Ii, Ici, Iasi, Ri, Rhi, Di, Iai, Hi, Ci, N_compartments, En, Im, Rhn, incH, incC
        state_indices = {}
        state_indices['En'] = self.model_struct["En"]
        state_indices['Im'] = self.model_struct["Im"]
        state_indices['Rhn'] = self.model_struct["Rhn"]
        state_indices['Si'] = 0
        state_indices['Ei'] = xp.array(state_indices['Si'] + 1 + xp.arange(state_indices['En']), dtype=int)
        state_indices['Ii'] = xp.array(state_indices['Ei'][-1] + 1 + xp.arange(state_indices['Im']), dtype=int)
        state_indices['Ici'] = xp.array(state_indices['Ii'][-1] + 1 + xp.arange(state_indices['Im']), dtype=int)
        state_indices['Iasi'] = xp.array(state_indices['Ici'][-1] + 1 + xp.arange(state_indices['Im']), dtype=int)
        state_indices['Ri'] = state_indices['Iasi'][-1] + 1
        state_indices['Rhi'] = xp.array(state_indices['Ri'] + 1 + xp.arange(state_indices['Rhn']), dtype=int)
        state_indices['Di'] = state_indices['Rhi'][-1] + 1

        state_indices['Iai'] = xp.hstack([state_indices['Ii'], state_indices['Iasi'], state_indices['Ici']])  # all I compartments
        state_indices['Hi'] = xp.hstack([state_indices['Rhi'], state_indices['Ici']])  # all compartments in hospitalization
        state_indices['Ci'] = xp.hstack([state_indices['Ii'], state_indices['Ici'], state_indices['Rhi']])

        state_indices['incH'] = state_indices['Di'] + 1
        state_indices['incC'] = state_indices['incH'] + 1

        state_indices['N_compartments'] = xp.to_cpu(state_indices['incC'] + 1)

        self.state_indices = state_indices

        self.get_state_indices()

    # We really need to refactor things so we don't ahve to do this...
    def get_state_indices(self):       
        for k in self.state_indices:
            globals()[k] = self.state_indices[k]

    def reset(self, seed=None, params=None):

        # if you set a seed using the constructor, you're stuck using it forever
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            xp.random.seed(seed)
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
            )  # TODO rename to inc_case_hist

            death_hist = xp.vstack(
                list(nx.get_node_attributes(G, "death_hist").values())
            ).T

            self.death_hist_cum = death_hist.astype(float)
            self.death_hist = xp.diff(death_hist, axis=0).astype(float)  # TODO rename

            if "IFR" in G.nodes[list(G.nodes.keys())[0]]:
                logging.info("Using ifr from graph")
                self.use_G_ifr = True
                node_IFR = nx.get_node_attributes(G, "IFR")
                self.ifr = xp.asarray((np.vstack(list(node_IFR.values()))).T)
            else:
                self.use_G_ifr = False

            # grab the geo id's for later
            self.adm2_id = np.fromiter(
                nx.get_node_attributes(G, G.graph["adm2_key"]).values(), dtype=int
            )

            # Mapping from index to adm1
            self.adm1_id = np.fromiter(
                nx.get_node_attributes(G, G.graph["adm1_key"]).values(), dtype=int
            )
            self.adm1_id = xp.asarray(self.adm1_id, dtype=np.int32)
            self.adm1_max = xp.to_cpu(self.adm1_id.max())

            # Make contact mats sym and normalized
            self.contact_mats = G.graph["contact_mats"]
            logging.debug(f"graph contact mats: {G.graph['contact_mats'].keys()}")
            for mat in self.contact_mats:
                c_mat = xp.array(self.contact_mats[mat])
                c_mat = (c_mat + c_mat.T) / 2.0
                self.contact_mats[mat] = c_mat
            # remove all_locations so we can sum over the them ourselves
            if "all_locations" in self.contact_mats:
                del self.contact_mats["all_locations"]

            self.Cij = xp.vstack(
                [self.contact_mats[k][None, ...] for k in sorted(self.contact_mats)]
            )

            # Get stratified population (and total)
            N_age_init = nx.get_node_attributes(G, "N_age_init")
            self.Nij = xp.asarray((np.vstack(list(N_age_init.values())) + 0.0001).T)
            self.Nj = xp.asarray(np.sum(self.Nij, axis=0))
            self.n_age_grps = self.Nij.shape[0]

            self.G = G
            n_nodes = len(self.G.nodes())

            self.first_date = datetime.date.fromisoformat(G.graph["start_date"])

            if args.npi_file is not None:
                logging.info(f"Using NPI from: {args.npi_file}")
                self.npi_params = read_npi_file(
                    args.npi_file,
                    self.first_date,
                    self.t_max,
                    self.adm2_id,
                    args.disable_npi,
                )
                for k in self.npi_params:
                    self.npi_params[k] = xp.array(self.npi_params[k])
                    if k == "contact_weights":
                        self.npi_params[k] = xp.broadcast_to(
                            self.npi_params[k], (self.t_max + 1, n_nodes, 4)
                        )
                    else:
                        self.npi_params[k] = xp.broadcast_to(
                            self.npi_params[k], (self.t_max + 1, n_nodes)
                        )
            else:
                self.npi_params = {
                    "r0_reduct": xp.broadcast_to(xp.ones(1), (self.t_max + 1, n_nodes)),
                    "contact_weights": xp.broadcast_to(
                        xp.ones(1), (self.t_max + 1, n_nodes, 4)
                    ),
                    "mobility_reduct": xp.broadcast_to(
                        xp.ones(1), (self.t_max + 1, n_nodes)
                    ),
                }

            self.Cij = xp.broadcast_to(self.Cij, (n_nodes,) + self.Cij.shape)
            self.npi_params["contact_weights"] = self.npi_params["contact_weights"][
                ..., None, None
            ]

            # Build adj mat for the RHS
            G = nx.convert_node_labels_to_integers(G)

            edges = xp.array(list(G.edges(data="weight"))).T

            A = sparse.coo_matrix(
                (edges[2], (edges[0].astype(int), edges[1].astype(int)))
            )
            A = A.toarray()

            self.baseline_A = A / xp.sum(A, axis=0)

            self.adm0_cfr_reported = None
            self.adm1_cfr_reported = None
            self.adm2_cfr_reported = None

            if "covid_tracking_data" in G.graph:
                self.rescale_chr = True
                ct_data = G.graph["covid_tracking_data"]
                ct_data.reset_index(inplace=True)
                hosp_data = ct_data.loc[ct_data.date == str(self.first_date)][
                    ["adm1", "hospitalizedCurrently"]
                ]
                hosp_data_adm1 = hosp_data["adm1"].to_numpy()
                hosp_data_count = hosp_data["hospitalizedCurrently"].to_numpy()
                self.adm1_current_hosp = xp.zeros((self.adm1_max + 1,), dtype=float)
                self.adm1_current_hosp[hosp_data_adm1] = hosp_data_count
                logging.debug("Current hosp: " + pformat(self.adm1_current_hosp))
                df = G.graph["covid_tracking_data"]
                self.adm1_current_cfr = xp.zeros((self.adm1_max + 1,), dtype=float)
                cfr_delay = 12

                for adm1, g in df.groupby("adm1"):
                    g_df = (
                        g.set_index("date")
                        .sort_index()
                        .rolling(7)
                        .mean()
                        .dropna(how="all")
                    )
                    g_df.clip(lower=0.0, inplace=True)
                    g_df = g_df.rolling(7).sum()
                    new_deaths = g_df.deathIncrease.to_numpy()
                    new_cases = g_df.positiveIncrease.to_numpy()
                    new_deaths = np.clip(new_deaths, a_min=0.0, a_max=None)
                    new_cases = np.clip(new_cases, a_min=0.0, a_max=None)
                    hist_cfr = new_deaths[cfr_delay:] / new_cases[:-cfr_delay]
                    cfr = np.nanmean(hist_cfr[-7:])
                    self.adm1_current_cfr[adm1] = cfr
                logging.debug("Current CFR: " + pformat(self.adm1_current_cfr))

            else:
                self.rescale_chr = False

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

        logging.debug("params: " + pformat(self.params, width=120))

        for k in self.params:
            if type(self.params[k]).__module__ == np.__name__:
                self.params[k] = xp.asarray(self.params[k])

        self.params.H = xp.broadcast_to(self.params.H[:, None], self.Nij.shape)
        self.params.F = xp.broadcast_to(self.params.F[:, None], self.Nij.shape)

        if self.use_G_ifr:
            self.ifr[xp.isnan(self.ifr)] = 0.0
            self.params.F = self.ifr / self.params["SYM_FRAC"]
            adm0_ifr = xp.sum(self.ifr * self.Nij) / xp.sum(self.Nj)
            ifr_scale = (
                0.0065 / adm0_ifr
            )  # TODO this should be in par file (its from planning scenario5)
            self.params.F = xp.clip(self.params.F * ifr_scale, 0.0, 1.0)
            self.params.F_old = self.params.F.copy()

            # TODO this needs to be cleaned up BAD
            # should add a util function to do the rollups to adm1 (it shows up in case_reporting/doubling t calc too)
            adm1_Fi = xp.zeros((self.adm1_max + 1, self.n_age_grps))
            xp.scatter_add(adm1_Fi, self.adm1_id, (self.params.F * self.Nij).T)
            adm1_Ni = xp.zeros((self.adm1_max + 1, self.n_age_grps))
            xp.scatter_add(adm1_Ni, self.adm1_id, self.Nij.T)
            adm1_Fi = adm1_Fi / adm1_Ni
            adm1_F = xp.mean(adm1_Fi, axis=1)

            adm1_F_fac = self.adm1_current_cfr / adm1_F
            adm1_F_fac[xp.isnan(adm1_F_fac)] = 1.0

            F_RR_fac = truncnorm(xp, 1.0, RR_VAR, size=adm1_F_fac.size, a_min=1e-6)
            adm1_F_fac = adm1_F_fac * F_RR_fac
            adm1_F_fac = xp.clip(
                adm1_F_fac, a_min=0.1, a_max=10.0
            )  # prevent extreme values
            logging.debug("adm1 cfr rescaling factor: " + pformat(adm1_F_fac))
            self.params.F = self.params.F * adm1_F_fac[self.adm1_id]
            self.params.F = 0.75 * xp.clip(self.params.F, a_min=1.0e-10, a_max=1.0)
            self.params.H = xp.clip(self.params.H, a_min=self.params.F, a_max=1.0)

        case_reporting = xp.to_cpu(
            self.estimate_reporting(cfr=self.params.F, days_back=22)
        )
        self.case_reporting = xp.array(
            mPERT_sample(
                mu=xp.clip(case_reporting, a_min=0.2, a_max=1.0),
                a=xp.clip(0.8 * case_reporting, a_min=0.2, a_max=None),
                b=xp.clip(1.2 * case_reporting, a_min=None, a_max=1.0),
                gamma=500.0,
            )
        )
        # self.case_reporting = self.estimate_reporting(cfr=self.params.F, days_back=22)

        self.doubling_t = self.estimate_doubling_time(mean_time_window=7)

        if xp.any(~xp.isfinite(self.doubling_t)):
            logging.info("non finite doubling times, is there enough case data?")
            logging.debug(self.doubling_t)
            logging.debug(self.adm1_id[~xp.isfinite(self.doubling_t)])
            raise SimulationException

        if RR_VAR > 0.0:
            self.doubling_t *= truncnorm(
                xp, 1.0, RR_VAR, size=self.doubling_t.shape, a_min=1e-6
            )
            self.doubling_t = xp.clip(self.doubling_t, 1.0, None)

        self.params = self.s_par.rescale_doubling_rate(
            self.doubling_t, self.params, xp, self.A
        )

        n_nodes = len(self.G.nodes())

        mean_case_reporting = xp.mean(self.case_reporting[-7:], axis=0)

        self.params["CASE_REPORT"] = mean_case_reporting
        self.params["THETA"] = xp.broadcast_to(
            self.params["THETA"][:, None], (self.n_age_grps, n_nodes)
        )
        self.params["GAMMA_H"] = xp.broadcast_to(
            self.params["GAMMA_H"][:, None], (self.n_age_grps, n_nodes)
        )
        self.params["F_eff"] = xp.clip(self.params["F"] / self.params["H"], 0.0, 1.0)

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

        R_fac = xp.array(mPERT_sample(mu=0.5, a=0.25, b=0.75, gamma=50.0))
        E_fac = xp.array(mPERT_sample(mu=1.6, a=1.35, b=1.85, gamma=50.0))
        H_fac = xp.array(mPERT_sample(mu=1.0, a=0.9, b=1.1, gamma=100.0))

        I_init = current_I[None, :] / self.Nij / self.n_age_grps
        D_init = self.init_deaths[None, :] / self.Nij / self.n_age_grps
        recovered_init = (
            (self.init_cum_cases)
            / self.params["SYM_FRAC"]
            / (self.params["CASE_REPORT"])
        ) * R_fac
        R_init = (
            (recovered_init) / self.Nij / self.n_age_grps
            - D_init
            - I_init / self.params["SYM_FRAC"]
        )  # rhi handled later

        self.params.H = self.params.H * H_fac

        ic_frac = 1.0 / (1.0 + self.params.THETA / self.params.GAMMA_H)
        hosp_frac = 1.0 / (1.0 + self.params.GAMMA_H / self.params.THETA)

        # print(ic_frac + hosp_frac)
        exp_frac = (
            E_fac
            * xp.ones(I_init.shape[-1])
            # * np.diag(self.A)
            # * np.sum(self.A, axis=1)
            * (self.params.R0)  # @ self.A)
            * self.params.GAMMA
            / self.params.SIGMA
        )

        y[Ii] = (1.0 - self.params.H) * I_init / len(Ii)
        # y[Ici] = ic_frac * self.params.H * I_init / (len(Ici))
        # y[Rhi] = hosp_frac * self.params.H * I_init / (Rhn)
        y[Ici] = self.params.CASE_REPORT * self.params.H * I_init / (len(Ici))
        y[Rhi] = (
            self.params.CASE_REPORT
            * self.params.H
            * I_init
            * self.params.GAMMA_H
            / self.params.THETA
            / Rhn
        )

        if self.rescale_chr:
            adm1_hosp = xp.zeros((self.adm1_max + 1,), dtype=float)
            xp.scatter_add(
                adm1_hosp, self.adm1_id, xp.sum(y[Hi] * self.Nij, axis=(0, 1))
            )
            adm2_hosp_frac = (self.adm1_current_hosp / adm1_hosp)[self.adm1_id]
            adm0_hosp_frac = xp.nansum(self.adm1_current_hosp) / xp.nansum(adm1_hosp)
            adm2_hosp_frac[xp.isnan(adm2_hosp_frac)] = adm0_hosp_frac
            self.params.H = xp.clip(
                H_fac * self.params.H * adm2_hosp_frac[None, :], self.params.F, 1.0
            )

            self.params["F_eff"] = xp.clip(
                self.params["F"] / self.params["H"], 0.0, 1.0
            )

            y[Ii] = (1.0 - self.params.H) * I_init / len(Ii)
            # y[Ici] = ic_frac * self.params.H * I_init / (len(Ici))
            # y[Rhi] = hosp_frac * self.params.H * I_init / (Rhn)
            y[Ici] = self.params.CASE_REPORT * self.params.H * I_init / (len(Ici))
            y[Rhi] = (
                self.params.CASE_REPORT
                * self.params.H
                * I_init
                * self.params.GAMMA_H
                / self.params.THETA
                / Rhn
            )

        y[Si] -= xp.sum(y[Ii], axis=0) + xp.sum(y[Ici], axis=0) + xp.sum(y[Rhi], axis=0)
        R_init -= xp.sum(y[Rhi], axis=0)

        y[Si] -= self.params.ASYM_FRAC / self.params.SYM_FRAC * I_init
        y[Iasi] = self.params.ASYM_FRAC / self.params.SYM_FRAC * I_init / len(Iasi)
        y[Si] -= exp_frac[None, :] * I_init
        y[Ei] = exp_frac[None, :] * I_init / len(Ei)
        y[Si] -= R_init
        y[Ri] = R_init
        y[Si] -= D_init
        y[Di] = D_init

        self.y = y

        # TODO assert this is 1. (need to take mean and around b/c fp err)
        # if xp.sum(self.y, axis=0)

        if xp.any(~xp.isfinite(self.y)):
            logging.info(
                "nonfinite values in the state vector, something is wrong with init"
            )
            raise SimulationException

        logging.debug("done reset")

        return y

    def reset_A(self, var):
        new_R0_fracij = truncnorm(xp, 1.0, var, size=self.A.shape, a_min=1e-6)
        new_R0_fracij = xp.clip(new_R0_fracij, 1e-6, None)
        A = self.baseline_A * new_R0_fracij
        self.A = A / xp.sum(A, axis=0)

    def estimate_doubling_time(
        self,
        days_back=14,
        doubling_time_window=7,
        mean_time_window=None,
        min_doubling_t=1.0,
    ):
        cases = self.case_hist_cum[-days_back:] / self.case_reporting[-days_back:]
        cases_old = (
            self.case_hist_cum[
                -days_back - doubling_time_window : -doubling_time_window
            ]
            / self.case_reporting[
                -days_back - doubling_time_window : -doubling_time_window
            ]
        )

        # adm0
        adm0_doubling_t = (
            doubling_time_window
            * xp.log(2.0)
            / xp.log(xp.nansum(cases, axis=1) / xp.nansum(cases_old, axis=1))
        )

        logging.debug("Adm0 doubling time: " + str(adm0_doubling_t))
        if xp.any(~xp.isfinite(adm0_doubling_t)):
            logging.debug(xp.nansum(cases, axis=1))
            logging.debug(xp.nansum(cases_old, axis=1))
            raise SimulationException

        doubling_t = xp.repeat(adm0_doubling_t[:, None], cases.shape[-1], axis=1)

        # adm1
        cases_adm1 = xp.zeros((self.adm1_max + 1, days_back), dtype=float)
        cases_old_adm1 = xp.zeros((self.adm1_max + 1, days_back), dtype=float)

        xp.scatter_add(cases_adm1, self.adm1_id, cases.T)
        xp.scatter_add(cases_old_adm1, self.adm1_id, cases_old.T)

        adm1_doubling_t = (
            doubling_time_window * xp.log(2.0) / xp.log(cases_adm1 / cases_old_adm1)
        )

        tmp_doubling_t = adm1_doubling_t[self.adm1_id].T
        valid_mask = xp.isfinite(tmp_doubling_t) & (tmp_doubling_t > min_doubling_t)

        doubling_t[valid_mask] = tmp_doubling_t[valid_mask]

        # adm2
        adm2_doubling_t = doubling_time_window * xp.log(2.0) / xp.log(cases / cases_old)

        valid_adm2_dt = xp.isfinite(adm2_doubling_t) & (
            adm2_doubling_t > min_doubling_t
        )
        doubling_t[valid_adm2_dt] = adm2_doubling_t[valid_adm2_dt]

        # hist_weights = xp.arange(1., days_back + 1.0, 1.0)
        # hist_doubling_t = xp.sum(doubling_t * hist_weights[:, None], axis=0) / xp.sum(
        #    hist_weights
        # )

        # Take mean of most recent values
        if mean_time_window is not None:
            hist_doubling_t = xp.nanmean(doubling_t[-mean_time_window:], axis=0)

        return hist_doubling_t

    def estimate_reporting(self, cfr, days_back=14, case_lag=None, min_deaths=100.0):

        if case_lag is None:
            adm0_cfr_by_age = xp.sum(cfr * self.Nij, axis=1) / xp.sum(self.Nj, axis=0)
            adm0_cfr_total = xp.sum(
                xp.sum(cfr * self.Nij, axis=1) / xp.sum(self.Nj, axis=0), axis=0,
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
            xp.sum(cfr * self.Nij, axis=1) / xp.sum(self.Nj, axis=0), axis=0
        )
        if self.adm0_cfr_reported is None:
            self.adm0_cfr_reported = xp.sum(
                self.death_hist_cum[-days_back:], axis=1
            ) / xp.sum(cases_lagged, axis=1)
        adm0_case_report = adm0_cfr_param / self.adm0_cfr_reported

        logging.debug("Adm0 case reporting rate: " + pformat(adm0_case_report))
        if xp.any(~xp.isfinite(adm0_case_report)):
            logging.debug("adm0 case report not finite")
            logging.debug(adm0_cfr_param)
            logging.debug(self.adm0_cfr_reported)
            raise SimulationException

        case_report = xp.repeat(
            adm0_case_report[:, None], cases_lagged.shape[-1], axis=1
        )

        # adm1
        adm1_cfr_param = xp.zeros((self.adm1_max + 1,), dtype=float)
        adm1_totpop = xp.zeros((self.adm1_max + 1,), dtype=float)

        tmp_adm1_cfr = xp.sum(cfr * self.Nij, axis=0)

        xp.scatter_add(adm1_cfr_param, self.adm1_id, tmp_adm1_cfr)
        xp.scatter_add(adm1_totpop, self.adm1_id, self.Nj)
        adm1_cfr_param /= adm1_totpop

        # adm1_cfr_reported is const, only calc it once and cache it
        if self.adm1_cfr_reported is None:
            self.adm1_deaths_reported = xp.zeros(
                (self.adm1_max + 1, days_back), dtype=float
            )
            adm1_lagged_cases = xp.zeros((self.adm1_max + 1, days_back), dtype=float)

            xp.scatter_add(
                self.adm1_deaths_reported,
                self.adm1_id,
                self.death_hist_cum[-days_back:].T,
            )
            xp.scatter_add(adm1_lagged_cases, self.adm1_id, cases_lagged.T)

            self.adm1_cfr_reported = self.adm1_deaths_reported / adm1_lagged_cases

        adm1_case_report = (adm1_cfr_param[:, None] / self.adm1_cfr_reported)[
            self.adm1_id
        ].T

        valid_mask = (self.adm1_deaths_reported > min_deaths)[
            self.adm1_id
        ].T & xp.isfinite(adm1_case_report)
        case_report[valid_mask] = adm1_case_report[valid_mask]

        # adm2
        adm2_cfr_param = xp.sum(cfr * (self.Nij / self.Nj), axis=0)

        if self.adm2_cfr_reported is None:
            self.adm2_cfr_reported = self.death_hist_cum[-days_back:] / cases_lagged
        adm2_case_report = adm2_cfr_param / self.adm2_cfr_reported

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
    def _dGdt_vec(t, y, Nij, contact_mats, Aij, par, npi):
        # constraint on values
        lower, upper = (0.0, 1.0)  # bounds for state vars

        # grab index of OOB values so we can zero derivatives (stability...)
        too_low = y <= lower
        too_high = y >= upper

        # reshape to a sane form (compartment, age, node)
        s = y.reshape(N_compartments, Nij.shape[0], -1)

        # Clip state to be in bounds (except allocs b/c thats a counter)
        xp.clip(s, a_min=lower, a_max=upper, out=s)

        # init d(state)/dt
        dG = xp.zeros(s.shape)

        # effective params after damping w/ allocated stuff
        BETA_eff = npi["r0_reduct"][int(t)] * par["BETA"]
        F_eff = par["F_eff"]
        H = par["H"]
        THETA = par["THETA"]
        GAMMA = par["GAMMA"]
        GAMMA_H = par["GAMMA_H"]
        SIGMA = par["SIGMA"]

        Cij = npi["contact_weights"][int(t)] * contact_mats
        Cij = xp.sum(Cij, axis=1)
        Cij /= xp.sum(Cij, axis=2)[..., None]

        Aij_eff = npi["mobility_reduct"][int(t)][..., None] * Aij

        # perturb Aij
        # new_R0_fracij = truncnorm(xp, 1.0, .1, size=Aij.shape, a_min=1e-6)
        # new_R0_fracij = xp.clip(new_R0_fracij, 1e-6, None)
        # A = Aij * new_R0_fracij
        # Aij_eff = A / xp.sum(A, axis=0)

        # Infectivity matrix (I made this name up, idk what its really called)
        I_tot = xp.sum(Nij * s[Iai], axis=0) - (1.0 - par["rel_inf_asym"]) * xp.sum(
            Nij * s[Iasi], axis=0
        )

        # I_tmp = (Aij.T @ I_tot.T).T
        I_tmp = I_tot @ Aij_eff  # using identity (A@B).T = B.T @ A.T

        beta_mat = s[Si] * xp.squeeze((Cij @ I_tmp.T[..., None]), axis=-1).T
        beta_mat /= Nij

        # dS/dt
        dG[Si] = -BETA_eff * (beta_mat)
        # dE/dt
        dG[Ei[0]] = BETA_eff * (beta_mat) - En * SIGMA * s[Ei[0]]
        for i in Ei[1:]:
            dG[i] = En * SIGMA * s[i - 1] - En * SIGMA * s[i]

        # dI/dt
        dG[Iasi[0]] = (
            par["ASYM_FRAC"] * En * SIGMA * s[Ei[-1]] - Im * GAMMA * s[Iasi[0]]
        )
        for i in Iasi[1:]:
            dG[i] = Im * GAMMA * s[i - 1] - Im * GAMMA * s[i]

        dG[Ii[0]] = (
            par["SYM_FRAC"] * (1.0 - H) * En * SIGMA * s[Ei[-1]] - Im * GAMMA * s[Ii[0]]
        )
        for i in Ii[1:]:
            dG[i] = Im * GAMMA * s[i - 1] - Im * GAMMA * s[i]

        # dIc/dt
        dG[Ici[0]] = (
            par["SYM_FRAC"] * H * En * SIGMA * s[Ei[-1]] - Im * GAMMA_H * s[Ici[0]]
        )
        for i in Ici[1:]:
            dG[i] = Im * GAMMA_H * s[i - 1] - Im * GAMMA_H * s[i]

        # dRhi/dt
        dG[Rhi[0]] = Im * GAMMA_H * s[Ici[-1]] - (Rhn * THETA) * s[Rhi[0]]
        for i in Rhi[1:]:
            dG[i] = Rhn * THETA * s[i - 1] - (Rhn * THETA) * s[i]

        # dR/dt
        dG[Ri] = (
            Im * GAMMA * (s[Ii[-1]] + s[Iasi[-1]])
            + (1.0 - F_eff) * Rhn * THETA * s[Rhi[-1]]
        )

        # dD/dt
        dG[Di] += F_eff * (THETA) * Rhn * s[Rhi[-1]]

        dG[incH] = par["SYM_FRAC"] * H * En * SIGMA * s[Ei[-1]]
        dG[incC] = par["SYM_FRAC"] * par["CASE_REPORT"] * En * SIGMA * s[Ei[-1]]

        # bring back to 1d for the ODE api
        dG = dG.reshape(-1)

        # zero derivatives for things we had to clip if they are going further out of bounds
        dG = xp.where(too_low & (dG < 0.0), 0.0, dG)
        dG = xp.where(too_high & (dG > 0.0), 0.0, dG)

        return dG

    def dGdt_vec(self, t, y, Nij, Cij, Aij, par):
        return self._dGdt_vec(t, y, Nij, Cij, Aij, par)

    def run_once(self, seed=None, outdir="raw_output/", output=True, output_queue=None):

        # reset everything
        logging.debug("Resetting state")
        self.reset(seed=seed)
        logging.debug("Done reset")

        # TODO should output the IC here

        # do integration
        logging.debug("Starting integration")
        t_eval = np.arange(0, self.t_max + self.dt, self.dt)
        sol = ivp.solve_ivp(
            self._dGdt_vec,
            method="RK23",
            t_span=(0.0, self.t_max),
            y0=self.y.reshape(-1),
            t_eval=t_eval,
            args=(self.Nij, self.Cij, self.A, self.params, self.npi_params),
        )
        logging.debug("Done integration")
        y = sol.y.reshape(N_compartments, self.n_age_grps, -1, len(t_eval))

        out = self.Nij[None, ..., None] * y

        # collapse age groups
        out = xp.sum(out, axis=1)

        population_conserved = (
            xp.diff(xp.around(xp.sum(out[:incH], axis=(0, 1)), 1)) == 0.0
        ).all()
        if not population_conserved:
            pass  # TODO we're getting small fp errors here
            # print(xp.sum(xp.diff(xp.around(xp.sum(out[:incH], axis=(0, 1)), 1))))
            # logging.error("Population not conserved!")
            # print(xp.sum(xp.sum(y[:incH],axis=0)-1.))
            # raise SimulationException

        adm2_ids = np.broadcast_to(self.adm2_id[:, None], out.shape[1:])

        n_time_steps = out.shape[-1]

        t_output = xp.to_cpu(sol.t)
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

        init_inc_death_mean = xp.mean(xp.sum(daily_deaths[:, 1:4], axis=0))
        hist_inc_death_mean = xp.mean(xp.sum(self.death_hist[-7:], axis=-1))

        inc_death_rejection_fac = 2.0  # 1.1
        if (init_inc_death_mean > inc_death_rejection_fac * hist_inc_death_mean) or (
            inc_death_rejection_fac * init_inc_death_mean < hist_inc_death_mean
        ):
            if REJECT_RUNS:
                logging.info("Inconsistent inc deaths, rejecting run")
                raise SimulationException

        cum_cases = (
            xp.sum(
                self.Nij[..., None] * (1.0 - xp.sum(y[: Ei[-1] + 1], axis=0)), axis=0
            )
            * self.params.SYM_FRAC
        )
        daily_cases = xp.diff(
            cum_cases,
            prepend=0,  # self.case_hist_cum[-1][:, None] / self.params.CASE_REPORT[:, None],
            axis=-1,
        )
        n_daily_cases = xp.diff(
            out[incC], axis=-1, prepend=self.case_hist_cum[-1][:, None]
        )
        n_cum_cases = self.case_hist_cum[-1][:, None] + out[incC]

        init_inc_case_mean = xp.mean(xp.sum(n_daily_cases[:, 1:4], axis=0))
        hist_inc_case_mean = xp.mean(xp.sum(self.case_hist[-7:], axis=-1))

        inc_case_rejection_fac = 2.0
        if (init_inc_case_mean > inc_case_rejection_fac * hist_inc_case_mean) or (
            inc_case_rejection_fac * init_inc_case_mean < hist_inc_case_mean
        ):
            if REJECT_RUNS:
                logging.info("Inconsistent inc cases, rejecting run")
                raise SimulationException

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
            "rid": np.broadcast_to(seed, out.shape[-1]).reshape(-1),
            "S": out[Si],
            "E": out[Ei],
            "I": out[Ii],
            "Ic": out[Ici],
            "Ia": out[Iasi],
            "R": out[Ri],
            "Rh": out[Rhi],
            "D": out[Di],
            "NH": xp.diff(out[incH], axis=-1, prepend=0.0),
            "NC": daily_cases.reshape(-1),
            "NCR": daily_cases_reported.reshape(-1),
            "ND": daily_deaths.reshape(-1),
            "CC": n_cum_cases.reshape(-1)
            / self.params.CASE_REPORT[:, None],  # cum_cases.reshape(-1),
            "CCR": n_cum_cases.reshape(-1),  # cum_cases_reported.reshape(-1),
            "ICU": xp.sum(icu, axis=0).reshape(-1),
            "VENT": xp.sum(vent, axis=0).reshape(-1),
            "nNCR": n_daily_cases.reshape(-1),
            "nCCR": n_cum_cases.reshape(-1),
            "CASE_REPORT": np.broadcast_to(
                self.params.CASE_REPORT[:, None], adm2_ids.shape
            ).reshape(-1),
            "Reff": (
                self.npi_params["r0_reduct"].T
                * np.broadcast_to(
                    (self.params.R0 * (np.diag(self.A)))[:, None], adm2_ids.shape
                )
            ).reshape(-1),
            "doubling_t": np.broadcast_to(
                self.doubling_t[:, None], adm2_ids.shape
            ).reshape(-1),
        }

        # Collapse the gamma-distributed compartments and move everything to cpu
        for k in df_data:
            if df_data[k].ndim == 2:
                df_data[k] = xp.sum(df_data[k], axis=0)

            df_data[k] = xp.to_cpu(df_data[k])

        # Append data to the hdf5 file
        output_folder = os.path.join(outdir, self.run_id)
        os.makedirs(output_folder, exist_ok=True)
        out_df = pd.DataFrame(data=df_data)

        # round off any FP error we picked up along the way
        out_df.set_index(["ADM2_ID", "date", "rid"], inplace=True)
        out_df = out_df.apply(np.around, args=(3,))
        out_df.reset_index(inplace=True)

        if output:
            # out_df.to_feather(os.path.join(output_folder, str(seed) + ".feather"))
            for date, date_df in out_df.groupby("date"):
                if output_queue is None:
                    date_df.reset_index().to_feather(
                        os.path.join(
                            output_folder,
                            str(seed) + "_" + str(date.date()) + ".feather",
                        )
                    )
                else:
                    output_queue.put(
                        (
                            os.path.join(
                                output_folder,
                                str(seed) + "_" + str(date.date()) + ".feather",
                            ),
                            date_df,
                        )
                    )

        # TODO we should output the per monte carlo param rolls, this got lost when we switched from hdf5


if __name__ == "__main__":

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    loglevel = 30 - 10 * min(args.verbosity, 2)
    runid = get_runid()
    if not os.path.exists(args.output_dir + "/" + runid):
        os.mkdir(args.output_dir + "/" + runid)
    fh = logging.FileHandler(args.output_dir + "/" + runid + "/stdout")
    fh.setLevel(logging.DEBUG)
    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s",
        handlers=[TqdmLoggingHandler()],
    )
    # TODO we should output the logs to output_dir too...
    _banner()

    import threading, queue

    to_write = queue.Queue()

    def writer():
        # Call to_write.get() until it returns None
        for fname, df in iter(to_write.get, None):
            df.reset_index().to_feather(fname)

    write_thread = threading.Thread(target=writer)
    write_thread.start()

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
    pbar = tqdm(total=n_mc, dynamic_ncols=True)
    try:
        while success < n_mc:
            start = datetime.datetime.now()
            try:
                with xp.optimize_kernels():
                    env.run_once(
                        seed=seed, outdir=args.output_dir, output_queue=to_write
                    )
                success += 1
                pbar.update(1)
            except SimulationException:
                pass
            seed += 1  # TODO add last seed to pbar
            run_time = (datetime.datetime.now() - start).total_seconds()
            times.append(run_time)

            logging.info(f"{seed}: {datetime.datetime.now() - start}")
    except (KeyboardInterrupt, SystemExit):
        logging.warning("Caught SIGINT, cleaning up")
        to_write.put(None)
        write_thread.join()
    finally:
        to_write.put(None)
        pbar.close()
        logging.info(f"Total runtime: {datetime.datetime.now() - total_start}")
