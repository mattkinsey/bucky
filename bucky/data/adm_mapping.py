from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from .._typing import ArrayLike, PathLike
from ..numerical_libs import sync_numerical_libs, xp
from ..util.cached_prop import cached_property


@dataclass(frozen=True)
class AdminLevel:
    level: int
    idx: ArrayLike
    ids: ArrayLike
    abbrs: ArrayLike
    names: ArrayLike

    @sync_numerical_libs
    def __init__(self, ids, level=0, abbrs=None, names=None):
        object.__setattr__(self, "level", level)
        uniq_ids, uniq_index, squashed_idxs = np.unique(ids, return_inverse=True, return_index=True)
        # TODO assert sizes make sense (in post init)?
        object.__setattr__(self, "ids", uniq_ids)
        object.__setattr__(self, "idx", squashed_idxs)
        if abbrs is None:
            abbrs = xp.to_cpu(ids).astype(str)
        # if abbr is just for uniq we need to do something like:
        # object.__setattr__(self, "abbrs", abbrs[xp.to_cpu(squashed_idxs)])
        object.__setattr__(self, "abbrs", abbrs[uniq_index])
        if names is None:
            names = xp.to_cpu(ids).astype(str)
        # object.__setattr__(self, "names", names[xp.to_cpu(squashed_idxs)])
        object.__setattr__(self, "names", names[uniq_index])

    def __repr__(self):
        # TODO say if we have valid names/abbrs?
        return f"Admin level {self.level}, with {len(self)} locations"

    def __len__(self):
        return len(self.ids)


# TODO allow initing with just adm1_ids
@dataclass(frozen=True)
class AdminLevelMapping:
    adm0: AdminLevel
    adm1: AdminLevel
    adm2: AdminLevel

    @sync_numerical_libs
    def __init__(
        self,
        adm2: AdminLevel,
        adm1: Optional[AdminLevel] = None,
        adm0: Optional[AdminLevel] = None,
        levels: dict = field(init=False),  # noqa: B008
    ):

        # TODO should set levels first then make attrs for every level we have automatically

        object.__setattr__(self, "adm2", adm2)

        if adm0 is None:
            # default to one country
            object.__setattr__(self, "adm0", AdminLevel(np.zeros(adm2.ids.shape, dtype=int), level=0))
        else:
            object.__setattr__(self, "adm0", adm0)

        if adm1 is None:
            # default to one state
            object.__setattr__(self, "adm1", AdminLevel(np.zeros(adm2.ids.shape, dtype=int), level=1))
        else:
            object.__setattr__(self, "adm1", adm1)

        object.__setattr__(self, "levels", {0: self.adm0, 1: self.adm1, 2: self.adm2})

    # def __post_init__(self):
    #    # some basic validation
    #    if len(self.adm1_ids) != len(self.adm2_ids):
    #        logger.error("Inconsistent adm1 and adm2 id used to create map.")
    #        raise ValueError
    #    if self.uniq_adm1_ids[self.adm1_ids] != self.actual_adm1_ids:
    #        logger.error("Squashed adm1 mapping failed to recover original ids.")
    #        raise ValueError

    def __repr__(self) -> str:
        return f"adm0 containing {len(self.adm1)} adm1 regions and {len(self.adm2)} adm2 regions"

    def mapping(self, from_: str, to: str, level: int):
        level_map = self.levels[level]
        if from_ in ("id", "name", "abbr"):
            from_ = f"{from_}s"
        if to in ("id", "name", "abbr"):
            to = f"{to}s"

        return {t: f for t, f in zip(getattr(level_map, from_), getattr(level_map, to))}

    @sync_numerical_libs
    def to_csv(self, filename: PathLike):
        data = {
            "adm2": self.adm2.ids[self.adm2.idx],
            "adm1": self.adm1.ids[self.adm1.idx],
            "adm0": self.adm0.ids[self.adm0.idx],
            "adm2_name": self.adm2.names[self.adm2.idx],
            "adm1_name": self.adm1.names[self.adm1.idx],
            "adm0_name": self.adm0.names[self.adm0.idx],
            "adm2_abbr": self.adm2.abbrs[self.adm2.idx],
            "adm1_abbr": self.adm1.abbrs[self.adm1.idx],
            "adm0_abbr": self.adm0.abbrs[self.adm0.idx],
        }

        pd.DataFrame(data).to_csv(filename, index=False)

    @staticmethod
    def from_csv(filename: PathLike):

        df = pd.read_csv(filename)
        df = df.sort_values(by=["adm2"])
        maps = {}
        for col in df.columns:
            maps[col] = np.array(df[col].to_numpy()) if pd.api.types.is_integer_dtype(df[col]) else df[col].to_numpy()

        return AdminLevelMapping(
            adm2=AdminLevel(level=2, ids=maps["adm2"], names=maps["adm2_name"], abbrs=maps["adm2_abbr"]),
            adm1=AdminLevel(level=1, ids=maps["adm1"], names=maps["adm1_name"], abbrs=maps["adm1_abbr"]),
            adm0=AdminLevel(level=0, ids=maps["adm0"], names=maps["adm0_name"], abbrs=maps["adm0_abbr"]),
        )

    @property
    def n_adm1(self):
        return len(self.adm1.ids)

    # some legacy stuff, we should warn on usage b/c its deprecated
    """
    @cached_property
    def adm1_ids(self):
        return self.adm1.idx

    @cached_property
    def uniq_adm1_ids(self):
        return self.adm1.ids

    """

    @cached_property
    def actual_adm1_ids(self):
        # TODO we should provide easier access to this...
        return self.adm1.ids[self.adm1.idx]
