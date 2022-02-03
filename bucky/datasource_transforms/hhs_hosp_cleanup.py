from pathlib import Path

import pandas as pd
import us
from IPython import embed


def transform(output_file):

    base_dir = Path().cwd()

    hhs_file = base_dir / "hhs_hosp.csv"

    df = pd.read_csv(hhs_file)

    state_abbr_map = us.states.mapping("abbr", "fips")

    df["adm1"] = df.state.map(state_abbr_map)
    df["date"] = pd.to_datetime(df["date"])

    df = df.set_index(["adm1", "date"]).sort_index()

    df["incident_hospitalizations"] = (
        df.previous_day_admission_adult_covid_confirmed + df.previous_day_admission_pediatric_covid_confirmed
    )

    df["current_hospitalizations"] = (
        df.total_adult_patients_hospitalized_confirmed_covid + df.total_pediatric_patients_hospitalized_confirmed_covid
    )

    df = df[["incident_hospitalizations", "current_hospitalizations"]]

    df = df.fillna(0)

    df.to_csv(output_file, index=True)

    # embed()
