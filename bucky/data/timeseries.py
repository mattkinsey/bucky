import datetime
from dataclasses import dataclass, field, fields, replace
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from .._typing import ArrayLike, PathLike
from ..numerical_libs import sync_numerical_libs, xp
from .adm_mapping import AdminLevel, AdminLevelMapping


# TODO now that this holds the adm_mapping, the adm_ids column can probably be replaced...
@dataclass(frozen=True)
class SpatialStratifiedTimeseries:
    adm_level: int
    adm_ids: ArrayLike
    dates: ArrayLike  # TODO cupy cant handle this...
    adm_mapping: AdminLevelMapping  # = field(init=False)

    def __post_init__(self):
        valid_shape = self.dates.shape + self.adm_ids.shape
        for f in fields(self):
            if "data_field" in f.metadata:
                field_shape = getattr(self, f.name).shape
                if field_shape != valid_shape:
                    logger.error("Invalid timeseries shape {}; expected {}.", field_shape, valid_shape)
                    raise ValueError

        # TODO handle other adm levels
        # if self.adm_level == 2:
        #    object.__setattr__(self, "adm_mapping", AdminLevelMapping(adm2=AdminLevel(xp.to_cpu(self.adm_ids))))

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
        force_enddate: Optional[datetime.date] = None,
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
        date_mask = _mask_date_range(dates, n_days, valid_date_range, force_enddate, force_enddate_dow)

        ret = {
            "dates": dates[date_mask],
            "adm_ids": xp.array(df.index.unique(level=adm_col).values),
        }

        for fcolumn, out_name in column_names.items():
            var_full_hist = xp.array(df[fcolumn].unstack().fillna(0.0).values).T
            ret[out_name] = var_full_hist[date_mask]

        return ret

    def to_dict(self, level=None):
        # get data to requested adm level
        obj = self.sum_adm_level(level) if level is not None else self

        ret = {f.name: getattr(obj, f.name) for f in fields(obj) if f.name not in ("adm_level", "adm_mapping")}

        # reshape index columns and get the right name for the adm id column
        ret_shp = ret["dates"].shape + ret["adm_ids"].shape
        ret["date"] = np.broadcast_to(ret.pop("dates")[..., None], ret_shp)
        adm_col_name = f"adm{obj.adm_level}"
        ret[adm_col_name] = np.broadcast_to(ret.pop("adm_ids")[None, ...], ret_shp)

        # Flatten arrays
        ret = {k: np.ravel(xp.to_cpu(v)) for k, v in ret.items()}

        return ret

    def to_dataframe(self, level=None):
        data_dict = self.to_dict(level)
        df = pd.DataFrame(data_dict)
        adm_col = df.columns[df.columns.str.match("adm[0-9]")].item()
        return df.set_index([adm_col, "date"]).sort_index()

    def to_csv(self, filename, level=None):
        # TODO log
        df = self.to_dataframe(level)
        df.to_csv(filename, index=True)

    def replace(self, **changes):
        return replace(self, **changes)

    def sum_adm_level(self, level: Union[int, str]):

        if type(level) == str:

            # Check string begins with 'adm'
            if level[:3] == "adm":
                level = int(level.split("adm")[-1])
            else:
                logger.error("String admin aggregation level must begin with adm")
                raise ValueError

        # TODO masking, weighting?
        if level > self.adm_level:
            logger.error("Requested sum to finer adm level than the data.")
            raise ValueError

        if level == self.adm_level:
            return self

        out_id_map = self.adm_mapping.levels[level].idx
        new_ids = self.adm_mapping.levels[level].ids

        new_data = {"adm_level": level, "adm_ids": new_ids, "dates": self.dates, "adm_mapping": self.adm_mapping}
        for f in fields(self):
            if "summable" in f.metadata:
                orig_ts = getattr(self, f.name)
                new_ts = xp.zeros(new_ids.shape + self.dates.shape, dtype=orig_ts.dtype)
                xp.scatter_add(new_ts, out_id_map, orig_ts.T)
                new_data[f.name] = new_ts.T

        return self.__class__(**new_data)

    def validate_isfinite(self):
        for f in fields(self):
            if "data_field" in f.metadata:
                col_name = f.name
                data = getattr(self, f.name)
                finite = xp.isfinite(data)
                if xp.any(~finite):
                    locs = xp.argwhere(~finite)
                    logger.error(
                        f"Nonfinite values found in column {col_name} of {self.__class__.__qualname__} at {locs}!",
                    )
                    raise RuntimeError


def _mask_date_range(
    dates: np.ndarray,
    n_days: Optional[int] = None,
    valid_date_range: Tuple[Optional[datetime.date], Optional[datetime.date]] = (None, None),
    force_enddate: Optional[datetime.date] = None,
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

    if force_enddate is not None:
        if force_enddate_dow is not None:
            if force_enddate.weekday() != force_enddate_dow:
                logger.error("Start date not consistant with required day of week")
                raise RuntimeError
        valid_date_range[1] = force_enddate

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
            object.__setattr__(self, "incident_deaths", xp.gradient(self.cumulative_deaths, axis=0, edge_order=2))

        super().__post_init__()

    @staticmethod
    def from_csv(
        file: PathLike,
        n_days: Optional[int] = None,
        valid_date_range=(None, None),
        force_enddate: Optional[datetime.date] = None,
        force_enddate_dow: Optional[int] = None,
        adm_mapping: Optional[AdminLevelMapping] = None,
    ):
        logger.info("Reading historical CSSE data from {}", file)
        adm_level = "adm2"
        var_dict = SpatialStratifiedTimeseries._generic_from_csv(
            file,
            n_days,
            valid_date_range,
            force_enddate,
            force_enddate_dow,
            adm_level,
            column_names={"cumulative_reported_cases": "cumulative_cases", "cumulative_deaths": "cumulative_deaths"},
        )

        var_dict["adm_mapping"] = adm_mapping
        return CSSEData(2, **var_dict)


@dataclass(frozen=True, repr=False)
class HHSData(SpatialStratifiedTimeseries):
    current_hospitalizations: ArrayLike = field(metadata={"data_field": True, "summable": True})
    incident_hospitalizations: ArrayLike = field(metadata={"data_field": True, "summable": True})

    # TODO we probably need to store a AdminLevelMapping in each timeseries b/c the hhs adm_ids dont line up with the csse ones after we aggregate them to adm1...
    @staticmethod
    def from_csv(
        file: PathLike,
        n_days: Optional[int] = None,
        valid_date_range=(None, None),
        force_enddate: Optional[datetime.date] = None,
        force_enddate_dow: Optional[int] = None,
        adm_mapping: Optional[AdminLevelMapping] = None,
    ):
        logger.info("Reading historical HHS hospitalization data from {}", file)
        adm_level = "adm1"
        var_dict = SpatialStratifiedTimeseries._generic_from_csv(
            file,
            n_days,
            valid_date_range,
            force_enddate,
            force_enddate_dow,
            adm_col=adm_level,
            column_names={
                "incident_hospitalizations": "incident_hospitalizations",
                "current_hospitalizations": "current_hospitalizations",
            },
        )

        var_dict["adm_mapping"] = adm_mapping
        return HHSData(1, **var_dict)


@dataclass(frozen=True, repr=False)
class BuckyFittedData(SpatialStratifiedTimeseries):
    cumulative_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
    cumulative_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})
    incident_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
    incident_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})


@dataclass(frozen=True, repr=False)
class BuckyFittedCaseData(SpatialStratifiedTimeseries):
    cumulative_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
    cumulative_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})
    incident_cases: ArrayLike = field(metadata={"data_field": True, "summable": True})
    incident_deaths: ArrayLike = field(metadata={"data_field": True, "summable": True})

    @staticmethod
    def from_csv(
        file: PathLike,
        n_days: Optional[int] = None,
        valid_date_range=(None, None),
        force_enddate: Optional[datetime.date] = None,
        force_enddate_dow: Optional[int] = None,
        adm_mapping: Optional[AdminLevelMapping] = None,
    ):
        logger.info("Reading historical CSSE data from {}", file)
        adm_level = "adm2"
        var_dict = SpatialStratifiedTimeseries._generic_from_csv(
            file,
            n_days,
            valid_date_range,
            force_enddate,
            force_enddate_dow,
            adm_level,
            column_names={
                "cumulative_cases": "cumulative_cases",
                "cumulative_deaths": "cumulative_deaths",
                "incident_cases": "incident_cases",
                "incident_deaths": "incident_deaths",
            },
        )

        var_dict["adm_mapping"] = adm_mapping
        return BuckyFittedCaseData(2, **var_dict)
