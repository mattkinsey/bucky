from pathlib import Path

import pandas as pd

# CSSE UIDs for Michigan prison information
MI_PRISON_UIDS = [84070004, 84070005]

# CSSE IDs for Utah local health districts
UT_LHD_UIDS = [84070015, 84070016, 84070017, 84070018, 84070019, 84070020]

nan_fips_uid_map = {
    84070002: 90025,
    84070005: 90026,
    84070004: 90026,
    84070003: 90029,
    84070015: 90049,
    84070016: 90049,
    84070017: 90049,
    84070018: 90049,
    84070019: 90049,
    84070020: 90049,
}


def transform(output_file, census_data_path):
    repo_base = Path().cwd()

    ts_base_path = repo_base / "csse_covid_19_data" / "csse_covid_19_time_series"

    case_file = ts_base_path / "time_series_covid19_confirmed_US.csv"
    death_file = ts_base_path / "time_series_covid19_deaths_US.csv"

    final_column_file_map = {
        "cumulative_reported_cases": case_file,
        "cumulative_deaths": death_file,
    }

    column_dfs = []
    for column_name, csv_file in final_column_file_map.items():

        # Add random bins with nan FIPS into their state unallocated
        df = pd.read_csv(csv_file)
        nan_fips_rows = df.loc[df["FIPS"].isna()]
        nan_fips_mapped_fips = nan_fips_rows["UID"].map(nan_fips_uid_map)
        df.loc[df["FIPS"].isna(), "FIPS"] = nan_fips_mapped_fips

        # aggregate duplicated FIPS values
        df = df.groupby("FIPS").sum()
        df.index = df.index.astype(int)

        # Drop cruise ships
        df = df.drop(index=[88888, 99999]).sort_index()

        # Move territories with only aggregate data to their unassigned bin
        df = df.reset_index()
        df.loc[df["FIPS"] < 1000, "FIPS"] = df.loc[df["FIPS"] < 1000, "FIPS"] + 90000

        # set adm1 index
        df["adm1"] = df["FIPS"] // 1000

        # census_data = repo_base.parent / "binned_census_age_groups.csv"
        census_df = pd.read_csv(census_data_path, index_col="adm2")
        adm2_pops = census_df.sum(axis=1).sort_index()
        census_df["adm1"] = census_df.index // 1000
        adm1_pops = census_df.groupby("adm1").sum().sum(axis=1)

        fraction_adm1_pop = adm2_pops / adm1_pops.loc[adm2_pops.index // 1000].astype(float).values

        # group and sum 'unassigned' and 'out of' fips for each state
        adm1_rows = df.loc[
            df["adm1"].isin(
                [
                    80,
                    90,
                ],
            )
        ].copy()
        df = df.loc[~df["adm1"].isin([80, 90])]
        df = df.set_index("FIPS").sort_index()  # drop from main df
        # adm
        adm1_rows["adm1"] = adm1_rows["FIPS"] % 1000
        adm1_rows = adm1_rows.groupby("adm1").sum()

        # fix indices
        uniq_adm1_index = (fraction_adm1_pop.index // 1000).unique().sort_values()
        adm1_rows = adm1_rows.reindex(uniq_adm1_index, fill_value=0.0)

        tmp = adm1_rows.loc[fraction_adm1_pop.index // 1000]
        tmp.index = fraction_adm1_pop.index
        tmp = tmp.mul(fraction_adm1_pop, axis=0)

        df = df.reindex(tmp.index, fill_value=0.0)
        df = df + tmp
        df = df.filter(regex="([0-9]+)/([0-9]+)/([0-9]+)")

        df = df.unstack().to_frame(column_name).reset_index().rename(columns={"level_0": "date"})
        df["date"] = pd.to_datetime(df["date"])
        column_dfs.append(df.set_index(["adm2", "date"]).sort_index())

    final_df = pd.concat(column_dfs, axis=1)
    final_df.to_csv(output_file, index=True)
