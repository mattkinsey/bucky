"""Global configuration handler for Bucky, also include prior parameters"""

from importlib import resources
from pathlib import Path

import yaml
from loguru import logger

from .numerical_libs import sync_numerical_libs, xp
from .util import distributions
from .util.extrapolate import interp_extrap
from .util.nested_dict import NestedDict


def locate_base_config():
    """Locate the base_config package that shipped with bucky (it's likely in site-packages)."""
    with resources.path("bucky", "base_config") as cfg_path:
        return Path(cfg_path)


def locate_current_config():
    """Find the config file/directory to use."""
    potential_locations = [
        Path.cwd(),
        Path.home(),
    ]

    for p in potential_locations:
        cfg_dir = p / "bucky.conf.d"
        if cfg_dir.exists() and cfg_dir.is_dir():
            logger.info("Using bucky config directory at {}", str(cfg_dir))
            return cfg_dir

        cfg_one_file = p / "bucky.yaml"
        if cfg_one_file.exists():
            logger.info("Using bucky config file at {}", str(cfg_one_file))
            return cfg_one_file

    base_cfg = locate_base_config()
    logger.warning("Local Bucky config not found, using defaults at {}", str(base_cfg))
    return base_cfg


class BuckyConfig(NestedDict):
    """Bucky configuration."""

    def load_cfg(self, par_path):
        """Read in the YAML cfg file(s)."""
        logger.info("Loading bucky config from {}", par_path)
        par = Path(par_path)

        try:
            if par.is_dir():
                for f in sorted(par.iterdir()):
                    self.update(yaml.safe_load(f.read_text(encoding="utf-8")))  # nosec
            else:
                self.update(yaml.safe_load(par.read_text(encoding="utf-8")))  # nosec
        except FileNotFoundError:
            logger.exception("Config not found!")

        self._convert_paths()
        # self._to_arrays()

        return self

    def _convert_paths(self):
        def _cast_to_path(v):
            ret = Path(v["path"])
            if not ret.exists():  # currently being created by old config.yml
                logger.warning("Path in cfg not found: {}", v["path"])
            return ret

        ret = self.apply(_cast_to_path, contains_filter="path")
        return ret

    @sync_numerical_libs
    def _to_arrays(self, copy=False):
        # wip
        def _cast_to_array(v):
            return v if isinstance(v, str) else xp.array(v)

        ret = self.apply(_cast_to_array, copy=copy)
        return ret

    @sync_numerical_libs
    def _to_lists(self, copy=False):
        # wip
        def _cast_to_list(v):
            return xp.to_cpu(xp.squeeze(v)).tolist() if isinstance(v, xp.ndarray) else v

        ret = self.apply(_cast_to_list, copy=copy)
        return ret

    def to_yaml(self, *args, **kwargs):
        return yaml.dump(self._to_lists(copy=True).to_dict(), *args, **kwargs)

    @staticmethod
    def _age_interp(x_bins_new, x_bins, y):
        """Interpolate parameters define in age groups to a new set of age groups."""
        # TODO we should probably account for population for the 65+ type bins...
        # TODO move
        x_bins_new = xp.array(x_bins_new)
        x_bins = xp.array(x_bins)
        if (x_bins_new.shape != x_bins.shape) or xp.any(x_bins_new != x_bins):
            x_mean_new = xp.mean(x_bins_new, axis=1)
            x_mean = xp.mean(x_bins, axis=1)
            return interp_extrap(x_mean_new, x_mean, y)
        return y

    @sync_numerical_libs
    def interp_age_bins(self):
        def _interp_one(d):
            d["value"] = self._age_interp(self["model.structure.age_bins"], d.pop("age_bins"), d["value"])
            return d

        ret = self.apply(_interp_one, contains_filter=["age_bins", "value"])
        return ret

    def promote_sampled_values(self):
        def _promote_values(d):
            return d["value"] if len(d) == 1 else d

        ret = self.apply(_promote_values, contains_filter="value")
        return ret

    @sync_numerical_libs
    def _set_default_variances(self, copy=False):
        def _set_reroll_var(d):
            if d["distribution.func"] == "truncnorm" and "scale" not in d["distribution"]:
                d["distribution.scale"] = xp.abs(
                    self["model.monte_carlo.reroll_variance"] * xp.array(d["distribution.loc"]),
                )
            return d

        ret = self.apply(_set_reroll_var, copy=copy, contains_filter="distribution")
        return ret

    # TODO move to own class like distributionalConfig?
    @sync_numerical_libs
    def sample_distributions(self):
        """Draw a sample from each distributional parameter and drop it inline (in a returned copy of self)"""

        # TODO add something like 'register_distribtions' so we dont have to iterate the tree to find them?
        def _sample_distribution(d):
            dist = d.pop("distribution")
            func = dist.pop("func")

            if hasattr(distributions, func):
                base_func = getattr(distributions, func)
            elif hasattr(xp.random, func):  # noqa: SIM106
                base_func = getattr(xp.random, func)
            else:
                raise ValueError(f"Distribution {func} does not exist!")

            d["value"] = base_func(**dist)
            return d

        # self._to_arrays()
        ret = self._set_default_variances(copy=True)
        # ret = ret._interp_age_bins()
        ret = ret.apply(_sample_distribution, contains_filter="distribution")
        ret = ret.interp_age_bins()
        ret = ret.promote_sampled_values()
        return ret


base_cfg = BuckyConfig()
cfg = BuckyConfig()

"""
def load_base_cfg(path):
    base_cfg.load_cfg(path)


def roll_cfg_distributions():
    cfg = base_cfg.sample_distributions()
"""

if __name__ == "__main__":
    file = "par2/"
    cfg = BuckyConfig().load_cfg(file)
    # print(cfg)

    samp = cfg.sample_distributions()
    # print(samp)
