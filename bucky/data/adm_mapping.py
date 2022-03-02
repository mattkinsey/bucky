from dataclasses import dataclass
from typing import Optional

from loguru import logger

from .._typing import ArrayLike, PathLike
from ..numerical_libs import sync_numerical_libs, xp


@dataclass(frozen=True)
class AdminLevelMapping:
    adm0: str
    adm1_ids: ArrayLike
    adm2_ids: ArrayLike
    uniq_adm1_ids: ArrayLike
    actual_adm1_ids: ArrayLike

    def __init__(self, adm0: str, adm2_ids: ArrayLike, adm1_ids: Optional[ArrayLike] = None):
        object.__setattr__(self, "adm0", adm0)
        object.__setattr__(self, "adm2_ids", adm2_ids)

        # get squashed mapping from adm2->adm1
        base_adm1_ids = self.adm2_ids // 1000 if adm1_ids is None else adm1_ids

        uniq_adm1_ids, squashed_adm1_ids = xp.unique(base_adm1_ids, return_inverse=True)
        object.__setattr__(self, "adm1_ids", squashed_adm1_ids)
        object.__setattr__(self, "uniq_adm1_ids", uniq_adm1_ids)
        object.__setattr__(self, "actual_adm1_ids", base_adm1_ids)

    def __post_init__(self):
        # some basic validation
        if len(self.adm1_ids) != len(self.adm2_ids):
            logger.error("Inconsistent adm1 and adm2 id used to create map.")
            raise ValueError
        if self.uniq_adm1_ids[self.adm1_ids] != self.actual_adm1_ids:
            logger.error("Squashed adm1 mapping failed to recover original ids.")
            raise ValueError

    def __repr__(self) -> str:
        return f"adm0 '{self.adm0}' containing {len(self.uniq_adm1_ids)} adm1 regions and {len(self.adm2_ids)} adm2 regions"

    def to_csv(self, filename: PathLike):
        raise NotImplementedError
