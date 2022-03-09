import datetime
from dataclasses import dataclass, field, fields, replace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .._typing import ArrayLike, PathLike
from ..numerical_libs import sync_numerical_libs, xp
from .adm_mapping import AdminLevelMapping


@dataclass(frozen=True)
class SpatialStratifiedTimeseries:
    adm_level: int
    adm_ids: ArrayLike
    dates: ArrayLike  # TODO cupy cant handle this...
    adm_mapping: AdminLevelMapping = field(init=False)

    def __post_init__(self):
        valid_shape = self.dates.shape + self.adm_ids.shape
        for f in fields(self):
            if "data_field" in f.metadata:
                field_shape = getattr(self, f.name).shape
                if field_shape != valid_shape:
                    logger.error("Invalid timeseries shape {}; expected {}.", field_shape, valid_shape)
                    raise ValueError

        # TODO handle other adm levels
        if self.adm_level == 2:
            object.__setattr__(self, "adm_mapping", AdminLevelMapping(adm0="US", adm2_ids=self.adm_ids))

    def __repr__(self) -> str:
        names = [f.name for f in fields(self) if f.name not in ["adm_ids", "dates"]]
        return (
            f"{names} for {self.adm_ids.shape[0]} adm{self.adm_level} regions from {self.start_date} to {self.end_date}"
        )

    @property
    def start_date(self) -> datetime.date:
        return self.dates[0]

    @property
    def end_date(self) -> datetime.date:
        return self.dates[-1]

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_loc(self) -> int:
        return len(self.adm_ids)

    @staticmethod
    @sync_numerical_libs
    def _generic_from_csv(
        filename: PathLike,
        n_days: Optional[int] = None,
        valid_date_range=(None, None),
        force_enddate_dow: Optional[int] = None,
        adm_col: str = "adm2",
        date_col: str = "date",
        column_names: Dict[str, str] = None,
    ):
        """Return a dict containing args to a subclass's constructor"""
        df = pd.read_csv(
            filename,
            index_col=[adm_col, date_col],
            engine="c",
            parse_dates=["date"],
        ).sort_index()

        dates = df.index.unique(level=date_col).values
        dates = dates.astype("datetime64[s]").astype(datetime.date)
        date_mask = _mask_date_range(dates, n_days, valid_date_range, force_enddate_dow)

        ret = {
            "dates": dates[date_mask],
            "adm_ids": df.index.unique(level=adm_col).values,
        }

        for fcolumn, out_name in column_names.items():
            var_full_hist = xp.array(df[fcolumn].unstack().fillna(0.0).values).T
            ret[out_name] = var_full_hist[date_mask]

        return ret

    def to_csv(self, filename):
        # TODO log
        output_dfs = {}
        for f in fields(self):
            if "data_field" in f.metadata:
                col_name = f.name
                data = getattr(self, f.name)
                col_df = pd.DataFrame(
                    xp.to_cpu(data),
                    index=xp.to_cpu(self.dates),
                    columns=xp.to_cpu(self.adm_ids),
                ).stack()
                output_dfs[col_name] = col_df

        pd.DataFrame(output_dfs).to_csv(filename, index_label=("date", "adm" + str(self.adm_level)))

    def replace(self, **changes):
        return replace(self, **changes)

    def sum_adm_level(self, level: int):
        # TODO masking, weighting?
        if level > self.adm_level:
            logger.error("Requested sum to finer adm level than the data.")
            raise ValueError

        if level == self.adm_level:
            return self

        if level == 1:
            out_id_map = self.adm_mapping.adm1_ids
            new_ids = self.adm_mapping.uniq_adm1_ids
        elif level == 0:
            out_id_map = xp.ones(self.adm_ids.shape, dtype=int)
            new_ids = xp.zeros((1,))
        else:
            raise NotImplementedError

        new_data = {"adm_level": level, "adm_ids": new_ids, "dates": self.dates}
        for f in fields(self):
            if "summable" in f.metadata:
                orig_ts = getattr(self, f.name)
                new_ts = xp.zeros(new_ids.shape + self.dates.shape, dtype=orig_ts.dtype)
                xp.scatter_add(new_ts, out_id_map, orig_ts.T)
                new_data[f.name] = new_ts.T

        return self.__class__(**new_data)


def _mask_date_range(
    dates: np.ndarray,
    n_days: Optional[int] = None,
    valid_date_range: Tuple[Optional[datetime.date], Optional[datetime.date]] = (None, None),
    force_enddate_dow: Optional[int] = None,
):
    valid_date_range = list(valid_date_range)

    if valid_date_range[0] is None:
        valid_date_range[0] = dates[0]
    if valid_date_range[1] is None:
        valid_date_range[1] = dates[-1]

    # Set the end of the date range to the last valid date that is the requested day of the week
    if force_enddate_dow is not None:
        end_date_dow = valid_date_range[1].weekday()
        days_after_forced_dow = (end_date_dow - force_enddate_dow + 7) % 7
        valid_date_range[1] = dates[-(days_after_forced_dow + 1)]

    # only grab the requested amount of history
    if n_days is not None:
        valid_date_range[0] = valid_date_range[-1] - datetime.timedelta(days=n_days - 1)

    # Mask out dates not in request range
    date_mask = (dates >= valid_date_range[0]) & (dates <= valid_date_range[1])
    return date_mask


@dataclass(frozen=True, repr=False)
class CSSEData(SpatialStratifiedTimeseries):
    cumulative_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
    cumulative_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})
    incident_cases: ArrayLike = field(default=None, metadata={"data_field": True, "summable": True})
    incident_deaths: ArrayLike = field(default=None, metadata={"data_field": True, "summable": True})

    def __post_init__(self):
        if self.incident_cases is None:
            object.__setattr__(self, "incident_cases", xp.gradient(self.cumulative_cases, axis=0, edge_order=2))
        if self.incident_deaths is None:
            object.__setattr__(self, "incident_deaths", xp.gradient(self.cumulative_cases, axis=0, edge_order=2))

        super().__post_init__()

    @staticmethod
    def from_csv(
        file: PathLike,
        n_days: Optional[int] = None,
        valid_date_range=(None, None),
        force_enddate_dow: Optional[int] = None,
    ):
        logger.info("Reading historical CSSE data from {}", file)
        adm_level = "adm2"
        var_dict = SpatialStratifiedTimeseries._generic_from_csv(
            file,
            n_days,
            valid_date_range,
            force_enddate_dow,
            adm_level,
            column_names={"cumulative_reported_cases": "cumulative_cases", "cumulative_deaths": "cumulative_deaths"},
        )
        return CSSEData(2, **var_dict)


@dataclass(frozen=True, repr=False)
class HHSData(SpatialStratifiedTimeseries):
    current_hospitalizations: ArrayLike = field(metadata={"data_field": True, "summable": True})
    incident_hospitalizations: ArrayLike = field(metadata={"data_field": True, "summable": True})

    # TODO we probably need to store a AdminLevelMapping in each timeseries b/c the hhs adm_ids dont line up with the csse ones after we aggregate them to adm1...
    @staticmethod
    def from_csv(file, n_days=None, valid_date_range=(None, None), force_enddate_dow=None):
        logger.info("Reading historical HHS hospitalization data from {}", file)
        adm_level = "adm1"
        var_dict = SpatialStratifiedTimeseries._generic_from_csv(
            file,
            n_days,
            valid_date_range,
            force_enddate_dow,
            adm_col=adm_level,
            column_names={
                "incident_hospitalizations": "incident_hospitalizations",
                "current_hospitalizations": "current_hospitalizations",
            },
        )

        return HHSData(1, **var_dict)


@dataclass(frozen=True, repr=False)
class BuckyFittedData(SpatialStratifiedTimeseries):
    cumulative_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
    cumulative_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})
    incident_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
    incident_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})
