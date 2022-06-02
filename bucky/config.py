"""Global configuration handler for Bucky, also include prior parameters"""

from glob import glob
from importlib import resources
from pathlib import Path, PosixPath

from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO
from ruamel.yaml.scalarfloat import ScalarFloat

yaml = YAML()


class YamlPath(PosixPath):
    yaml_tag = "!path"

    @classmethod
    def to_yaml(cls, representer, node):
        # print(f"{''.join(node.parts)}")
        return representer.represent_scalar(cls.yaml_tag, f"{'/'.join(node.parts)}")

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(node.value)


yaml.register_class(YamlPath)
from loguru import logger

from .numerical_libs import sync_numerical_libs, xp
from .util import distributions
from .util.extrapolate import interp_extrap
from .util.nested_dict import NestedDict


def locate_base_config():
    """Locate the base_config package that shipped with bucky (it's likely in site-packages)."""
    with resources.path("bucky", "__init__.py") as pkg_init_path:
        return pkg_init_path.parent / "base_config"


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

        cfg_one_file = p / "bucky.yml"
        if cfg_one_file.exists():
            logger.info("Using bucky config file at {}", str(cfg_one_file))
            return cfg_one_file

    base_cfg = locate_base_config()
    logger.warning("Local Bucky config not found, using defaults at {}", str(base_cfg))
    return base_cfg


class BuckyConfig(NestedDict):
    """Bucky configuration."""

    def load_cfg(self, par_path):
        base_cfg = locate_base_config()

        self._load_one_cfg(base_cfg)
        if par_path != base_cfg:
            self._load_one_cfg(par_path)

        self._cast_floats()
        return self

    def _load_one_cfg(self, par_path):
        """Read in the YAML cfg file(s)."""
        logger.info("Loading bucky config from {}", par_path)
        par = Path(par_path)

        try:
            if par.is_dir():
                for f_str in sorted(glob(str(par / "**"), recursive=True)):
                    f = Path(f_str)
                    if f.is_file():
                        if f.suffix not in {".yml", ".yaml"}:
                            logger.warning("Ignoring non YAML file {}", f)
                            continue

                        logger.debug("Loading config file {}", f)
                        self.update(yaml.load(f.read_text(encoding="utf-8")))  # nosec
            else:
                self.update(yaml.load(par.read_text(encoding="utf-8")))  # nosec
        except FileNotFoundError:
            logger.exception("Config not found!")

        return self

    @sync_numerical_libs
    def _to_arrays(self, copy=False):
        # wip
        def _cast_to_array(v):
            return v if isinstance(v, str) else xp.array(v)

        ret = self.apply(_cast_to_array, copy=copy, apply_to_lists=True)
        return ret

    @sync_numerical_libs
    def _to_lists(self, copy=False):
        # wip
        def _cast_to_list(v):
            return xp.to_cpu(xp.squeeze(v)).tolist() if isinstance(v, xp.ndarray) else v

        ret = self.apply(_cast_to_list, copy=copy)
        return ret

    def _cast_floats(self, copy=False):
        def _cast_float(v):
            return float(v) if isinstance(v, ScalarFloat) else v

        ret = self.apply(_cast_float, copy=copy)
        return ret

    def to_yaml(self, *args, **kwargs):
        stream = StringIO()

        yaml.dump(self._to_lists(copy=True).to_dict(), stream, *args, **kwargs)
        return stream.getvalue()

    @staticmethod
    def _age_interp(x_bins_new, x_bins, y):
        """Interpolate parameters define in age groups to a new set of age groups."""
        # TODO we should probably account for population for the 65+ type bins...
        # TODO move
        x_bins_new = xp.array(x_bins_new)
        x_bins = xp.array(x_bins)
        y = xp.array(y)
        if (x_bins_new.shape != x_bins.shape) or xp.any(x_bins_new != x_bins):
            x_mean_new = xp.mean(x_bins_new, axis=1)
            x_mean = xp.mean(x_bins, axis=1)
            return interp_extrap(x_mean_new, x_mean, y)
        return y

    @sync_numerical_libs
    def interp_age_bins(self):
        def _interp_values_one(d):
            d["value"] = self._age_interp(self["model.structure.age_bins"], d.pop("age_bins"), d["value"])
            return d

        def _interp_dists_one(d):
            bins = d.pop("age_bins")
            if "loc" in d["distribution"]:
                d["distribution.loc"] = self._age_interp(self["model.structure.age_bins"], bins, d["distribution.loc"])
            if "scale" in d["distribution"]:
                d["distribution.scale"] = self._age_interp(
                    self["model.structure.age_bins"],
                    bins,
                    d["distribution.scale"],
                )
            return d

        self._to_arrays()
        ret = self.apply(_interp_values_one, contains_filter=["age_bins", "value"])
        ret = ret.apply(_interp_dists_one, contains_filter=["age_bins", "distribution"])
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
                    xp.array(self["model.monte_carlo.default_gaussian_variance"]) * xp.array(d["distribution.loc"]),
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
            dist = d.pop("distribution")._to_arrays()
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
        ret = ret.interp_age_bins()
        ret = ret.apply(_sample_distribution, contains_filter="distribution")
        ret = ret.interp_age_bins()
        ret = ret.promote_sampled_values()
        ret = ret._cast_floats()
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
