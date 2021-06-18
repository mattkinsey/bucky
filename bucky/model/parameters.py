"""Submodule to handle the model parameterization and randomization."""
import logging
from functools import partial
from os import listdir, path

import yaml

from ..numerical_libs import sync_numerical_libs, xp
from ..util import distributions, dotdict
from ..util.distributions import generic_distribution


def calc_Te(Tg, Ts, n, f):
    """Calculate the latent period."""
    num = 2.0 * n * f / (n + 1.0) * Tg - Ts
    den = 2.0 * n * f / (n + 1.0) - 1.0
    return num / den


def calc_Reff(m, n, Tg, Te, r):
    """Calculate the effective reproductive number."""
    num = 2.0 * n * r / (n + 1.0) * (Tg - Te) * (1.0 + r * Te / m) ** m
    den = 1.0 - (1.0 + 2.0 * r / (n + 1.0) * (Tg - Te)) ** (-n)
    return num / den


def calc_Ti(Te, Tg, n):
    """Calcuate the infectious period."""
    return (Tg - Te) * 2.0 * n / (n + 1.0)


def calc_beta(Te):
    """Derive beta from Te."""
    return 1.0 / Te


def calc_gamma(Ti):
    """Derive gamma from Ti."""
    return 1.0 / Ti


def CI_to_std(CI):
    """Convert a 95% confidence interval to an equivilent stddev (assuming its normal)."""
    lower, upper = CI
    std95 = xp.sqrt(1.0 / 0.05)
    return (upper + lower) / 2.0, (upper - lower) / std95 / 2.0


# TODO move to util
def recursive_dict_update(d, u):
    """Recursive update() for nested dicts"""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class buckyParams:
    """Class holding all the model parameters defined in the par file, also used to reroll them for each MC run."""

    @sync_numerical_libs
    def __init__(self, par_file=None):
        """Initialize the class, sync up the libs with the parent context and load the par file."""

        # TODO add flags that are read from the yaml
        self.base_params = dotdict({})
        self.param_funcs = dotdict({})
        self.consts = dotdict({})

        if par_file is not None:
            self.update_params_from_file(par_file)

    def update_params_from_file(self, par_file):
        """Update parameter distributions and consts from new yaml file."""
        self.base_params = self.read_yml(par_file)
        self.update_params(self.base_params)

    def update_params(self, update_dict):
        """Update parameter distributions and consts from nested dicts."""
        self.consts = recursive_dict_update(self.consts, {k: xp.array(v) for k, v in update_dict["consts"].items()})
        self.base_params = recursive_dict_update(self.base_params, update_dict)
        self._generate_param_funcs(self.base_params)

    @staticmethod
    def read_yml(par_file):
        """Read in the YAML par file"""
        # TODO check file exists
        # If par_file is a directory, read files in alphanumeric order
        if path.isdir(par_file):
            root = par_file
            files = listdir(par_file)
            files.sort()
        else:
            root = ""
            files = [par_file]
        # Read in first parameter file
        with open(path.join(root, files[0]), "rb") as f:
            d = yaml.safe_load(f)  # nosec
        # Update dictionary with additional parameter files
        for filename in files[1:]:
            with open(path.join(root, filename), "rb") as f:
                dp = yaml.safe_load(f)
                d = recursive_dict_update(d, dp)
        return d

    def generate_params(self):
        """Generate a new set of params by rerolling, adding the derived params and rejecting invalid sets."""
        while True:  # WTB python do-while...
            params = self.reroll_params()
            if self.consts.Te_min < params.Te < params.Tg and params.Ti > self.consts.Ti_min:
                params.consts = self.consts
                return params
            # logging.debug("Rejected params: " + pformat(params))

    def _generate_param_funcs(self, base_params):
        """Generate all the partial functions to roll values of the params."""
        # Default standard deviation (as percentage of mean)
        var = self.consts.reroll_variance if "reroll_variance" in self.consts else 0.2
        # Existing parameter functions
        param_funcs = self.param_funcs

        for p, params in base_params.items():
            # Collect distribution name
            if "dist" not in params:
                continue
            dist = params.pop("dist")

            # Set default scale if none is specified (could find better method for this)
            if dist == "truncnorm" and "scale" not in params:
                params["scale"] = xp.abs(var * xp.array(params["loc"]))

            # Clip function
            clip = None
            if "clip" in params:
                clip = partial(xp.clip, **params.pop("clip"))
            # Need to clip after interpolation
            elif "age_bins" in params:
                a_min = params.get("a_min")
                a_max = params.get("a_max")
                if a_min is not None or a_max is not None:
                    clip = partial(xp.clip, a_min=a_min, a_max=a_max)

            # Interpolate function
            interp = None
            if "age_bins" in params:
                standard_age_bins = xp.array(base_params["consts"]["age_bins"])
                age_bins = xp.array(params.pop("age_bins"))
                interp = partial(self.age_interp, x_bins_new=standard_age_bins, x_bins=age_bins)

            # Main distribution function
            if hasattr(distributions, dist):
                base_func = getattr(distributions, dist)
            elif hasattr(xp.random, dist):
                base_func = getattr(xp.random, dist)
            else:
                logging.error("Distribution {} does not exist!".format(dist))
                continue

            params = {k: xp.array(v) for k, v in params.items()}

            param_funcs[p] = partial(generic_distribution, base_func=base_func, params=params, interp=interp, clip=clip)

    def reroll_params(self):
        """Sample each parameter from distribution and calculate derived parameters."""
        return self.calc_derived_params(dotdict({p: f() for p, f in self.param_funcs.items()}))

    @staticmethod
    def age_interp(x_bins_new, x_bins, y):
        """Interpolate parameters define in age groups to a new set of age groups."""
        # TODO we should probably account for population for the 65+ type bins...
        x_bins_new = xp.array(x_bins_new)
        x_bins = xp.array(x_bins)
        if (x_bins_new.shape != x_bins.shape) or xp.any(x_bins_new != x_bins):
            x_mean_new = xp.mean(x_bins_new, axis=1)
            x_mean = xp.mean(x_bins, axis=1)
            return xp.interp(x_mean_new, x_mean, y)
        return y

    def calc_derived_params(self, params):
        """Add the derived params that are calculated from the rerolled ones."""
        En = self.consts.En
        Im = self.consts.Im
        params["Te"] = calc_Te(
            params["Tg"],
            params["Ts"],
            En,
            params["frac_trans_before_sym"],
        )
        params["Ti"] = calc_Ti(params["Te"], params["Tg"], En)
        r = xp.log(2.0) / params["D"]
        params["R0"] = calc_Reff(
            Im,
            En,
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
