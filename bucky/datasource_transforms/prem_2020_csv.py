from pathlib import Path

import pandas as pd


def transform(output_filename):

    base_dir = Path().cwd()

    df = pd.read_csv(base_dir / "prem_2020_contact_matrices.csv", engine="c")

    # grab US 'overall' values
    # TODO can we differentiate rural and urban counties?
    # See https://www.medrxiv.org/content/10.1101/2020.07.22.20159772v2.full.pdf
    df = df.loc[(df["iso3c"] == "USA") & (df["setting"] == "overall")]

    df = df.rename(
        columns={
            "location_contact": "location",
            "age_contactor": "i",
            "age_cotactee": "j",
            "mean_number_of_contacts": "value",
        },
    )
    df = df[["location", "i", "j", "value"]]

    df["i"] = pd.factorize(df["i"])[0]
    df["j"] = pd.factorize(df["j"])[0]

    df = df.set_index(["location", "i", "j"]).sort_index()

    df.to_csv(output_filename, index=True)
