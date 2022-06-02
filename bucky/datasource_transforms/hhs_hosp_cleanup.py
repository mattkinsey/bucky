from pathlib import Path

import pandas as pd
import us

# TODO grab other columns like icu usage?


def transform(output_file):

    base_dir = Path().cwd()

    hhs_file = base_dir / "hhs_hosp.csv"

    df = pd.read_csv(hhs_file)

    state_abbr_map = us.states.mapping("abbr", "fips")

    df["adm1"] = df.state.map(state_abbr_map)
    df["date"] = pd.to_datetime(df["date"])

    df = df.set_index(["adm1", "date"]).sort_index()

    inc_hosps = df.previous_day_admission_adult_covid_confirmed + df.previous_day_admission_pediatric_covid_confirmed

    current_hosps = (
        df.total_adult_patients_hospitalized_confirmed_covid + df.total_pediatric_patients_hospitalized_confirmed_covid
    )

    df = pd.concat({"incident_hospitalizations": inc_hosps, "current_hospitalizations": current_hosps}, axis=1)

    # Add in nan rows for the missing territories
    if 66 in df.index.unique("adm1"):
        raise RuntimeError("Unexpected adm1 66 in HHS Data")
    if 69 in df.index.unique("adm1"):
        raise RuntimeError("Unexpected adm1 69 in HHS Data")
    empty_row_inds = pd.MultiIndex.from_product([[66, 69], df.index.unique("date")])
    df = df.reindex(df.index.append(empty_row_inds)).sort_index()

    df = df.fillna(0)

    df.to_csv(output_file, index=True)
