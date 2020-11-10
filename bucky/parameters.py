"""TODO: Summary
"""
import copy
import logging
from pprint import pformat

import numpy as np
import yaml

from .util import dotdict
from .util.distributions import mPERT_sample, truncnorm


def calc_Te(Tg, Ts, n, f):
    """TODO: Summary

    Parameters
    ----------
    Tg : TYPE
        TODO: Description
    Ts : TYPE
        TODO: Description
    n : TYPE
        TODO: Description
    f : TYPE
        TODO: Description

    Returns
    -------
    TYPE
        TODO: Description
    """
    num = 2.0 * n * f / (n + 1.0) * Tg - Ts
    den = 2.0 * n * f / (n + 1.0) - 1.0
    return num / den


def calc_Reff(m, n, Tg, Te, r):
    """TODO: Summary

    Parameters
    ----------
    m : TYPE
        TODO: Description
    n : TYPE
        TODO: Description
    Tg : TYPE
        TODO: Description
    Te : TYPE
        TODO: Description
    r : TYPE
        TODO: Description

    Returns
    -------
    TYPE
        TODO: Description
    """
    num = 2.0 * n * r / (n + 1.0) * (Tg - Te) * (1.0 + r * Te / m) ** m
    den = 1.0 - (1.0 + 2.0 * r / (n + 1.0) * (Tg - Te)) ** (-n)
    return num / den


def calc_Ti(Te, Tg, n):
    """TODO: Summary

    Parameters
    ----------
    Te : TYPE
        TODO: Description
    Tg : TYPE
        TODO: Description
    n : TYPE
        TODO: Description

    Returns
    -------
    TYPE
        TODO: Description
    """
    return (Tg - Te) * 2.0 * n / (n + 1.0)


def calc_beta(Te):
    """TODO: Summary

    Parameters
    ----------
    Te : TYPE
        TODO: Description

    Returns
    -------
    TYPE
        TODO: Description
    """
    return 1.0 / Te


def calc_gamma(Ti):
    """TODO: Summary

    Parameters
    ----------
    Ti : TYPE
        TODO: Description

    Returns
    -------
    TYPE
        TODO: Description
    """
    return 1.0 / Ti


def CI_to_std(CI):
    """TODO: Summary

    Parameters
    ----------
    CI : TYPE
        TODO: Description

    Returns
    -------
    TYPE
        TODO: Description
    """
    lower, upper = CI
    std95 = np.sqrt(1.0 / 0.05)
    return (upper + lower) / 2.0, (upper - lower) / std95 / 2.0


class buckyParams:

    """TODO: Summary

    Attributes
    ----------
    base_params : TYPE
    TODO: Description
    consts : TYPE
    TODO: Description
    par_file : TYPE
    TODO: Description
    """

    def __init__(self, par_file=None):
        """TODO: Summary

        Parameters
        ----------
        par_file : None, optional
            TODO: Description
        """
        self.par_file = par_file
        if par_file is not None:
            self.base_params = self.read_yml(par_file)
            self.consts = dotdict(self.base_params["consts"])
        else:
            self.base_params = None

    @staticmethod
    def read_yml(par_file):
        """TODO: Summary

        Parameters
        ----------
        par_file : TYPE
            TODO: Description

        Returns
        -------
        TYPE
            TODO: Description
        """
        # TODO check file exists
        with open(par_file, "rb") as f:
            return yaml.load(f, yaml.SafeLoader)  # nosec

    def generate_params(self, var=0.2):
        """TODO: Summary

        Parameters
        ----------
        var : float, optional
            TODO: Description

        Returns
        -------
        TYPE
            TODO: Description
        """
        if var is None:
            var = 0.0
        while True:  # WTB python do-while...
            params = self.reroll_params(self.base_params, var)
            params = self.calc_derived_params(params)
            if (params.Te > 1.0 and params.Tg > params.Te and params.Ti > 1.0) or var == 0.0:
                return params
            logging.debug("Rejected params: " + pformat(params))

    def reroll_params(self, base_params, var):
        """TODO: Summary

        Parameters
        ----------
        base_params : TYPE
            TODO: Description
        var : TYPE
            TODO: Description

        Returns
        -------
        TYPE
            TODO: Description
        """
        params = dotdict({})
        for p in base_params:
            # Scalars
            if "gamma" in base_params[p]:
                mu = copy.deepcopy(base_params[p]["mean"])
                params[p] = mPERT_sample(np.array([mu]), gamma=base_params[p]["gamma"])

            elif "mean" in base_params[p]:
                if "CI" in base_params[p]:
                    if var:
                        params[p] = truncnorm(np, *CI_to_std(base_params[p]["CI"]), a_min=1e-6)
                    else:  # just use mean if we set var to 0
                        params[p] = copy.deepcopy(base_params[p]["mean"])
                else:
                    params[p] = copy.deepcopy(base_params[p]["mean"])
                    params[p] *= truncnorm(np, loc=1.0, scale=var, a_min=1e-6)

            # age-based vectors
            elif "values" in base_params[p]:
                params[p] = np.array(base_params[p]["values"])
                params[p] *= truncnorm(np, 1.0, var, size=params[p].shape, a_min=1e-6)
                # interp to our age bins
                if base_params[p]["age_bins"] != base_params["consts"]["age_bins"]:
                    params[p] = self.age_interp(
                        base_params["consts"]["age_bins"],
                        base_params[p]["age_bins"],
                        params[p],
                    )

            # fixed values (noop)
            else:
                params[p] = copy.deepcopy(base_params[p])

            # clip values
            if "clip" in base_params[p]:
                clip_range = base_params[p]["clip"]
                params[p] = np.clip(params[p], clip_range[0], clip_range[1])

        return params

    @staticmethod
    def age_interp(x_bins_new, x_bins, y):  # TODO we should probably account for population for the 65+ type bins...
        """TODO: Summary

        Parameters
        ----------
        x_bins_new : TYPE
            TODO: Description
        x_bins : TYPE
            TODO: Description
        y : TYPE
            TODO: Description

        Returns
        -------
        TYPE
            TODO: Description
        """
        x_mean_new = np.mean(np.array(x_bins_new), axis=1)
        x_mean = np.mean(np.array(x_bins), axis=1)
        return np.interp(x_mean_new, x_mean, y)

    @staticmethod
    def rescale_doubling_rate(D, params, xp, A_diag=None):
        """TODO: Summary

        Parameters
        ----------
        D : TYPE
            TODO: Description
        params : TYPE
            TODO: Description
        xp : TYPE
            TODO: Description
        A_diag : None, optional
            TODO: Description

        Returns
        -------
        TYPE
            TODO: Description
        """
        # TODO rename D to Td everwhere for consistency
        r = xp.log(2.0) / D
        params["R0"] = calc_Reff(
            params["consts"]["Im"],
            params["consts"]["En"],
            params["Tg"],
            params["Te"],
            r,
        )
        params["BETA"] = params["R0"] * params["GAMMA"]
        if A_diag is not None:
            # params['BETA'] /= xp.sum(A,axis=1)
            params["BETA"] /= A_diag
        return params

    @staticmethod
    def calc_derived_params(params):
        """TODO: Summary

        Parameters
        ----------
        params : TYPE
            TODO: Description

        Returns
        -------
        TYPE
            TODO: Description
        """
        params["Te"] = calc_Te(
            params["Tg"],
            params["Ts"],
            params["consts"]["En"],
            params["frac_trans_before_sym"],
        )
        params["Ti"] = calc_Ti(params["Te"], params["Tg"], params["consts"]["En"])
        r = np.log(2.0) / params["D"]
        params["R0"] = calc_Reff(
            params["consts"]["Im"],
            params["consts"]["En"],
            params["Tg"],
            params["Te"],
            r,
        )

        params["SIGMA"] = 1.0 / params["Te"]
        params["GAMMA"] = 1.0 / params["Ti"]
        params["BETA"] = params["R0"] * params["GAMMA"]
        params["SYM_FRAC"] = 1.0 - params["ASYM_FRAC"]
        params["THETA"] = 1.0 / params["H_TIME"]
        params["GAMMA_H"] = 1.0 / params["I_TO_H_TIME"]
        return params
