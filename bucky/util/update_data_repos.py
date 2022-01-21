"""Data Updating Utility (:mod:`bucky.util.update_data_repos`).

A utility for fetching updated data for mobility and case data from public repositories.

This module pulls from public git repositories and preprocessed the
data if necessary. For case data, unallocated or unassigned cases are
distributed as necessary.

"""
import logging
import os
import ssl
import subprocess
import urllib.request

import numpy as np
import pandas as pd
import tqdm

from .read_config import bucky_cfg

# Options for correcting territory data
TERRITORY_DATA = bucky_cfg["data_dir"] + "/population/territory_pop.csv"
ADD_AMERICAN_SAMOA = False

# CSSE UIDs for Michigan prison information
MI_PRISON_UIDS = [84070004, 84070005]

# CSSE IDs for Utah local health districts
UT_LHD_UIDS = [84070015, 84070016, 84070017, 84070018, 84070019, 84070020]


def get_timeseries_data(col_name, filename, fips_key="FIPS", is_csse=True):
    """Transforms a historical data file to a dataframe with FIPs, date, and case or death data.

    Parameters
    ----------
    col_name : str
        Column name to extract from data.
    filename : str
        Location of filename to read.
    fips_key : str, optional
        Key used in file for indicating county-level field.
    is_csse : bool, optional
        Indicates whether the file is CSSE data. If True, certain areas
        without FIPS are included.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with the historical data indexed by FIPS, date

    """
    # Read file
    df = pd.read_csv(filename)

    # CSSE-specific correction
    if is_csse:
        # Michigan prisons have no FIPS, replace with their UID to be processed later
        mi_data = df.loc[df["UID"].isin(MI_PRISON_UIDS)]
        mi_data = mi_data.assign(FIPS=mi_data["UID"])

        df.loc[mi_data.index] = mi_data.values  # noqa: PD011

        # Utah health districts have NAN FIPS, replace with UID
        utah_local_dist_data = df.loc[df["UID"].isin(UT_LHD_UIDS)]
        utah_local_dist_data = utah_local_dist_data.assign(FIPS=utah_local_dist_data["UID"])
        df.loc[utah_local_dist_data.index] = utah_local_dist_data.values

    # Get dates and FIPS columns only
    cols = list(df.columns)
    idx = cols.index("1/22/20")

    # Keep columns after index
    keep_cols = cols[idx:]

    # Add FIPS
    keep_cols.append(fips_key)

    # Drop other columns
    df = df[keep_cols]

    # Reindex and stack
    df = df.set_index(fips_key)

    # Stack
    df = df.stack().reset_index()

    # Replace column names
    df.columns = ["FIPS", "date", col_name]

    return df


def distribute_unallocated_csse(confirmed_file, deaths_file, hist_df):
    """Distributes unallocated historical case and deaths data from CSSE.

    JHU CSSE data contains state-level unallocated data, indicated with
    "Unassigned" or "Out of" for each state. This function distributes
    these unallocated cases based on the proportion of cases in each
    county relative to the state.

    Parameters
    ----------
    confirmed_file : str
        filename of CSSE confirmed data
    deaths_file : str
        filename of CSSE death data
    hist_df : pandas.DataFrame
        current historical DataFrame containing confirmed and death data
        indexed by date and FIPS code

    Returns
    -------
    hist_df : pandas.DataFrame
        modified historical DataFrame with cases and deaths distributed

    """
    hist_df = hist_df.reset_index()
    if "index" in hist_df.columns:
        hist_df = hist_df.drop(columns=["index"])

    hist_df = hist_df.assign(state_fips=hist_df["FIPS"] // 1000)
    hist_df = hist_df.set_index(["date", "FIPS"])
    # Read cases and deaths files
    case_df = pd.read_csv(confirmed_file)
    deaths_df = pd.read_csv(deaths_file)

    # Get unassigned and 'out of X'
    cases_unallocated = case_df.loc[
        (case_df["Combined_Key"].str.contains("Out of")) | (case_df["Combined_Key"].str.contains("Unassigned"))
    ]
    cases_unallocated = cases_unallocated.assign(state_fips=cases_unallocated["FIPS"].astype(str).str[3:].astype(float))

    deaths_unallocated = deaths_df.loc[
        (deaths_df["Combined_Key"].str.contains("Out of")) | (deaths_df["Combined_Key"].str.contains("Unassigned"))
    ]
    deaths_unallocated = deaths_unallocated.assign(
        state_fips=deaths_unallocated["FIPS"].astype(str).str[3:].astype(float),
    )

    # Sum unassigned and 'out of X'
    extra_cases = cases_unallocated.groupby("state_fips").sum()
    extra_deaths = deaths_unallocated.groupby("state_fips").sum()
    extra_cases = extra_cases.drop(
        columns=[
            "UID",
            "code3",
            "FIPS",
            "Lat",
            "Long_",
        ],
    )
    extra_deaths = extra_deaths.drop(
        columns=[
            "UID",
            "Population",
            "code3",
            "FIPS",
            "Lat",
            "Long_",
        ],
    )

    # Reformat dates to match processed data's format
    extra_cases.columns = pd.to_datetime(extra_cases.columns)
    extra_deaths.columns = pd.to_datetime(extra_deaths.columns)

    # Iterate over states in historical data
    for state_fips in tqdm.tqdm(
        extra_cases.index.array,
        desc="Distributing unallocated state data",
        dynamic_ncols=True,
    ):

        # Get extra cases and deaths
        state_extra_cases = extra_cases.xs(state_fips)
        state_extra_deaths = extra_deaths.xs(state_fips)

        # Get historical data
        state_df = hist_df.loc[hist_df["state_fips"] == state_fips]
        state_df = state_df.reset_index()
        state_confirmed = state_df[["FIPS", "date", "cumulative_reported_cases"]]

        state_confirmed = state_confirmed.pivot(index="FIPS", columns="date", values="cumulative_reported_cases")
        frac_df = state_confirmed / state_confirmed.sum()
        frac_df = frac_df.replace(np.nan, 0)

        # Distribute cases and deaths based on this matrix
        dist_cases = frac_df.mul(state_extra_cases, axis="columns").T.stack()
        dist_deaths = frac_df.mul(state_extra_deaths, axis="columns").T.stack()

        # Index historical data
        state_df = state_df.set_index(["date", "FIPS"])
        tmp = dist_deaths.to_frame(name="cumulative_deaths")
        tmp["cumulative_reported_cases"] = dist_cases
        state_df += tmp

        hist_df.loc[state_df.index] = state_df.values

    hist_df = hist_df.drop(columns=["state_fips"])
    return hist_df


def distribute_data_by_population(total_df, dist_vect, data_to_dist, replace):
    """Distributes data by population across a state or territory.

    Parameters
    ----------
    total_df : pandas.DataFrame
        DataFrame containing confirmed and death data indexed by date and
        FIPS code
    dist_vect : pandas.DataFrame
        Population data for each county as proportion of total state
        population, indexed by FIPS code
    data_to_dist: pandas.DataFrame
        Data to distribute, indexed by data
    replace : bool
        If true, distributed values overwrite current historical data in
        DataFrame. If false, distributed values are added to current data


    Returns
    -------
    total_df : pandas.DataFrame
        Modified input dataframe with distributed data

    """
    # Create temporary dataframe and merge
    tmp = total_df.reset_index()
    tmp = tmp.merge(dist_vect, on="FIPS")
    tmp = tmp.merge(data_to_dist, on="date")

    # Use population fraction to scale
    if replace:
        tmp = tmp.assign(cumulative_reported_cases=tmp["pop_fraction"] * tmp["cumulative_reported_cases_y"])
        tmp = tmp.assign(cumulative_deaths=tmp["pop_fraction"] * tmp["cumulative_deaths_y"])
    else:
        tmp = tmp.assign(
            cumulative_reported_cases=tmp["cumulative_reported_cases_x"]
            + tmp["pop_fraction"] * tmp["cumulative_reported_cases_y"],
        )
        tmp = tmp.assign(
            cumulative_deaths=tmp["cumulative_deaths_x"] + tmp["pop_fraction"] * tmp["cumulative_deaths_y"],
        )

    # Discard merge columns
    tmp = tmp[["FIPS", "date", "cumulative_reported_cases", "cumulative_deaths"]]
    tmp = tmp.set_index(["FIPS", "date"])
    total_df.loc[tmp.index] = tmp.values

    return total_df


def get_county_population_data(csse_deaths_file, county_fips):
    """Uses JHU CSSE deaths file to get county population data as as fraction of population across list of counties.

    Parameters
    ----------
    csse_deaths_file : str
        filename of CSSE deaths file
    county_fips: numpy.ndarray
        list of FIPS to return population data for


    Returns
    -------
    population_df: pandas.DataFrame
        DataFrame with population fraction data indexed by FIPS

    """
    # Use CSSE Deaths file to get population values by FIPS
    df = pd.read_csv(csse_deaths_file)
    population_df = df.loc[df["FIPS"].isin(county_fips)][["FIPS", "Population"]].set_index("FIPS")
    population_df = population_df.assign(pop_fraction=population_df["Population"] / population_df["Population"].sum())
    population_df = population_df.drop(columns=["Population"])

    return population_df


def distribute_utah_data(df, csse_deaths_file):
    """Distributes Utah case data for local health departments spanning multiple counties.

    Utah has 13 local health districts, six of which span multiple counties. This
    function distributes those cases and deaths by population across their constituent
    counties.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing historical data indexed by FIPS and date
    csse_deaths_file : str
        File location of CSSE deaths file

    Returns
    -------
    df : pandas.DataFrame
        Modified DataFrame containing corrected Utah historical data
        indexed by FIPS and date
    """
    local_districts = {
        # Box Elder, Cache, Rich
        84070015: {"name": "Bear River, Utah, US", "FIPS": [49003, 49005, 49033]},
        # Juab, Millard, Piute, Sevier, Wayne, Sanpete
        84070016: {"name": "Central Utah, Utah, US", "FIPS": [49023, 49027, 49031, 49041, 49055, 49039]},
        # Carbon, Emery, Grand
        84070017: {"name": "Southeast Utah, Utah, US", "FIPS": [49007, 49015, 49019]},
        # Garfield, Iron, Kane, Washington, Beaver
        84070018: {"name": "Southwest Utah, Utah, US", "FIPS": [49017, 49021, 49025, 49053, 49001]},
        # Daggett, Duchesne, Uintah
        84070019: {"name": "TriCounty, Utah, Utah, US", "FIPS": [49009, 49013, 49047]},
        # Weber, Morgan
        84070020: {"name": "Weber-Morgan, Utah, US", "FIPS": [49057, 49029]},
    }

    for district_uid, local_district in local_districts.items():

        # Get list of fips
        fips_list = local_district["FIPS"]

        # Deaths file has population data
        county_pop = get_county_population_data(csse_deaths_file, fips_list)

        # Get district data
        district_data = df.loc[district_uid]

        # Add to Michigan data, do not replace
        df = distribute_data_by_population(df, county_pop, district_data, True)

    # Drop health districts data from dataframe
    df = df.loc[~df.index.get_level_values(0).isin(UT_LHD_UIDS)]

    return df


def distribute_nyc_data(df):
    """Distributes NYC case data across the six NYC counties.

    TODO add deprecation warning b/c csse has fixed this

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing historical data indexed by FIPS and date

    Returns
    -------
    df : pandas.DataFrame
        Modified DataFrame containing corrected NYC historical data
        indexed by FIPS and date

    """
    # Get population for counties
    # nyc_counties = [36005, 36081, 36047, 36085, 36061]

    # CSSE has incorrect population data
    county_populations = {
        36005: 1432132,
        36081: 2278906,
        36047: 2582830,
        36085: 476179,
        36061: 1628701,
    }

    total_nyc_pop = np.sum(list(county_populations.values()))
    population_df = pd.DataFrame(
        data=county_populations.values(),
        columns=["Population"],
        index=county_populations.keys(),
    )
    population_df = population_df.assign(pop_fraction=population_df["Population"] / total_nyc_pop)
    population_df = population_df.drop(columns=["Population"])
    population_df.index = population_df.index.set_names(["FIPS"])

    # All data is in FIPS 36061
    nyc_data = df.xs(36061, level=0)

    df = distribute_data_by_population(df, population_df, nyc_data, True)
    return df


def distribute_mdoc(df, csse_deaths_file):
    """Distributes Michigan Department of Corrections data across Michigan counties by population.

    Parameters
    ----------
    df : pandas.DataFrame
        Current historical DataFrame indexed by FIPS and date, which
        includes MDOC and FCI data
    csse_deaths_file : str
        File location of CSSE deaths file (contains population data)

    Returns
    -------
    df : pandas.DataFrame
        Modified historical dataframe with Michigan prison data distributed
        and added to Michigan data

    """
    # Get Michigan county populations
    tmp = df.reset_index()
    michigan_counties = tmp.loc[tmp["FIPS"] // 1000 == 26]["FIPS"].unique()
    michigan_pop = get_county_population_data(csse_deaths_file, michigan_counties)

    # Get prison data
    mdoc_data = df.xs(MI_PRISON_UIDS[0], level=0)
    fci_data = df.xs(MI_PRISON_UIDS[1], level=0)

    # Sum and distribute
    mi_unallocated = mdoc_data + fci_data

    # Add to Michigan data, do not replace
    df = distribute_data_by_population(df, michigan_pop, mi_unallocated, False)

    # Drop prison data from dataframe
    df = df.loc[~df.index.get_level_values(0).isin(MI_PRISON_UIDS)]

    return df


def distribute_territory_data(df, add_american_samoa):
    """Distributes territory-wide case and death data for territories.

    Uses county population to distribute cases for US Virgin Islands,
    Guam, and CNMI. Optionally adds a single case to the most populous
    American Samoan county.

    Parameters
    ----------
    df : pandas.DataFrame
        Current historical DataFrame indexed by FIPS and date, which
        includes territory-wide case and death data
    add_american_samoa: bool
        If true, adds 1 case to American Samoa

    Returns
    -------
    df : pandas.DataFrame
        Modified historical dataframe with territory-wide data
        distributed to counties

    """
    # Get population data from file
    age_df = pd.read_csv(TERRITORY_DATA, index_col="fips")

    # use age-stratified data to get total pop per county
    pop_df = pd.DataFrame(age_df.sum(axis=1)).reset_index()
    pop_df = pop_df.rename(columns={"fips": "FIPS", 0: "total"})

    # Drop PR because CSSE does have county-level PR data now
    pop_df = pop_df.loc[~(pop_df["FIPS"] // 1000).isin([72, 60])]

    # Create nan dataframe for territories (easier to update than append)
    tfips = pop_df["FIPS"].unique()
    dates = df.index.unique(level=1).array
    fips_col = []
    date_col = []
    for fips in tfips:
        for d in dates:
            fips_col.append(fips)
            date_col.append(d)

    tframe = pd.DataFrame.from_dict(
        {
            "FIPS": fips_col,
            "date": date_col,
            "cumulative_reported_cases": [np.nan for d in date_col],
            "cumulative_deaths": [np.nan for d in date_col],
        },
    ).set_index(["FIPS", "date"])
    df = df.append(tframe)

    # CSSE has state-level data for Guam, CNMI, USVI
    state_level_fips = [66, 69, 78]

    for state_fips in state_level_fips:
        state_data = df.xs(state_fips, level=0)
        state_pop = pop_df.loc[pop_df["FIPS"] // 1000 == state_fips]
        state_pop = state_pop.assign(pop_fraction=state_pop["total"] / state_pop["total"].sum())
        state_pop = state_pop.drop(columns=["total"])
        df = distribute_data_by_population(df, state_pop, state_data, True)

    # Optionally add 1 confirmed case to most populous AS county
    if add_american_samoa:
        as_frame = pd.DataFrame.from_dict(
            {
                "FIPS": [60050 for d in dates],
                "date": dates,
                "cumulative_reported_cases": [1.0 for d in dates],
                "cumulative_deaths": [0.0 for d in dates],
            },
        ).set_index(["FIPS", "date"])
        df = df.append(as_frame)

    return df


def process_csse_data():
    """Performs pre-processing on CSSE data.

    CSSE data is separated into two different files: confirmed cases and
    deaths. These two files are combined into one dataframe, indexed by
    FIPS and date with two columns, Confirmed and Deaths. This function
    distributes CSSE that is either unallocated or territory-wide instead
    of county-wide. Michigan data from the state Department of Corrections
    and Federal Correctional Institution is distributed to Michigan counties.
    New York City data which is currently all placed in one county (New
    York County) is distributed to the other NYC counties. Territory data
    for Guam, CNMI, and US Virgin Islands is also distributed. This data
    is written to a CSV.

    """
    data_dir = bucky_cfg["data_dir"] + "/cases/COVID-19/csse_covid_19_data/csse_covid_19_time_series/"

    # Get confirmed and deaths files
    confirmed_file = os.path.join(data_dir, "time_series_covid19_confirmed_US.csv")
    deaths_file = os.path.join(data_dir, "time_series_covid19_deaths_US.csv")

    confirmed = get_timeseries_data("Confirmed", confirmed_file)
    deaths = get_timeseries_data("Deaths", deaths_file)

    # rename columns
    confirmed = confirmed.rename(columns={"Confirmed": "cumulative_reported_cases"})
    deaths = deaths.rename(columns={"Deaths": "cumulative_deaths"})

    # Merge datasets
    data = confirmed.merge(deaths, on=["FIPS", "date"], how="left").fillna(0)

    # Remove missing FIPS
    data = data[data.FIPS != 0]

    # Replace FIPS with adm2
    # data.rename(columns={"FIPS" : "adm2"}, inplace=True)
    # print(data.columns)

    data = data.set_index(["FIPS", "date"])

    # Distribute territory and Michigan DOC data
    data = distribute_territory_data(data, ADD_AMERICAN_SAMOA)
    data = distribute_mdoc(data, deaths_file)
    data = distribute_utah_data(data, deaths_file)

    data = data.reset_index()
    data = data.assign(date=pd.to_datetime(data["date"]))
    data = data.sort_values(by="date")

    data = distribute_unallocated_csse(confirmed_file, deaths_file, data)

    # Rename FIPS index to adm2
    data.index = data.index.rename(["date", "adm2"])

    # Write to files
    hist_file = bucky_cfg["data_dir"] + "/cases/csse_hist_timeseries.csv"
    logging.info(f"Saving CSSE historical data as {hist_file}")
    data.to_csv(hist_file)


def update_covid_tracking_data():
    """Downloads and processes data from the COVID Tracking project to match the format of other preprocessed data.

    The COVID Tracking project contains data at a state-level. Each state
    is given a random FIPS selected from all FIPS in that state. This is
    done to make aggregation easier for plotting later. Processed data is
    written to a CSV.

    """
    url = "https://api.covidtracking.com/v1/states/daily.csv"
    filename = bucky_cfg["data_dir"] + "/cases/covid_tracking_raw.csv"
    # Download data
    context = ssl._create_unverified_context()  # pylint: disable=W0212  # nosec
    # Create filename
    with urllib.request.urlopen(url, context=context) as testfile, open(filename, "w", encoding="utf-8") as f:  # nosec
        f.write(testfile.read().decode())

    # Read file
    data_file = bucky_cfg["data_dir"] + "/cases/covid_tracking_raw.csv"
    df = pd.read_csv(data_file)

    # Fix date
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    # Rename FIPS
    df = df.rename(columns={"fips": "adm1"})

    # Save
    covid_tracking_name = bucky_cfg["data_dir"] + "/cases/covid_tracking.csv"
    logging.info(f"Saving COVID Tracking Data as {covid_tracking_name}")
    df.to_csv(covid_tracking_name, index=False)


def process_usafacts(case_file, deaths_file):
    """Performs preprocessing on USA Facts data.

    USAFacts contains unallocated cases and deaths for each state. These
    are allocated across states based on case distribution in the state.

    Parameters
    ----------
    case_file : str
        Location of USAFacts case file
    deaths_file : str
        Location of USAFacts death file

    Returns
    -------
    combined_df : pandas.DataFrame
        USAFacts data containing cases and deaths indexed by FIPS and
        date.

    """
    # Read case file, will be used to scale unallocated cases & deaths
    case_df = pd.read_csv(case_file)
    case_df = case_df.drop(columns=["County Name", "State", "stateFIPS"])

    files = [case_file, deaths_file]
    cols = ["Confirmed", "Deaths"]
    processed_frames = []
    for i, file in enumerate(files):

        df = pd.read_csv(file)
        df = df.drop(columns=["County Name", "State"])

        # Get time series versions
        ts = get_timeseries_data(cols[i], file, "countyFIPS", False)

        ts = ts.loc[~ts["FIPS"].isin([0, 1])]
        ts = ts.set_index(["FIPS", "date"])

        for state_code, state_df in tqdm.tqdm(
            df.groupby("stateFIPS"),
            desc="Processing USAFacts " + cols[i],
            dynamic_ncols=True,
        ):

            # DC has no unallocated row
            if state_df.loc[state_df["countyFIPS"] == 0].empty:
                continue

            state_df = state_df.set_index("countyFIPS")
            state_df = state_df.drop(columns=["stateFIPS"])

            # Get unallocated cases
            state_unallocated = state_df.xs(0)
            state_df = state_df.drop(index=0)

            if state_code == 36:

                # NYC includes "probable" deaths - these are currently being dropped
                state_df = state_df.drop(index=1)

            if state_unallocated.sum() > 0:

                # Distribute by cases
                state_cases = case_df.loc[case_df["countyFIPS"] // 1000 == state_code]
                state_cases = state_cases.set_index("countyFIPS")
                frac_df = state_cases / state_cases.sum()
                frac_df = frac_df.replace(np.nan, 0)

                # Multiply with unallocated to distribute
                dist_unallocated = frac_df.mul(state_unallocated, axis="columns").T
                state_df = state_df.T + dist_unallocated
                state_df = state_df.T.stack().reset_index()

                # Replace column names
                state_df = state_df.rename(columns={"countyFIPS": "FIPS", "level_1": "date", 0: cols[i]})
                state_df = state_df.set_index(["FIPS", "date"])
                ts.loc[state_df.index] = state_df.values

        processed_frames.append(ts)

    # Combine
    combined_df = processed_frames[0].merge(processed_frames[1], on=["FIPS", "date"], how="left").fillna(0)
    return combined_df


def update_usafacts_data():
    """Retrieves updated historical data from USA Facts, preprocesses it, and writes to CSV."""

    logging.info("Downloading USA Facts data")
    case_url = "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv"
    deaths_url = "https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv"
    urls = [case_url, deaths_url]

    filenames = [
        bucky_cfg["data_dir"] + "/cases/covid_confirmed_usafacts.csv",
        bucky_cfg["data_dir"] + "/cases/covid_deaths_usafacts.csv",
    ]

    # Download case and death data
    context = ssl._create_unverified_context()  # pylint: disable=W0212  # nosec
    for i, url in enumerate(urls):

        # Create filename
        with urllib.request.urlopen(url, context=context) as testfile, open(  # nosec
            filenames[i],
            "w",
            encoding="utf-8",
        ) as f:
            f.write(testfile.read().decode())

    # Merge datasets
    data = process_usafacts(filenames[0], filenames[1])
    data = data.reset_index()

    # Sort by date
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(by="date")
    data = data.rename(columns={"FIPS": "adm2"})
    data.to_csv(bucky_cfg["data_dir"] + "/cases/usafacts_hist.csv")


def update_hhs_hosp_data():
    """Retrieves updated historical data from healthdata.gov and writes to CSV."""

    logging.info("Downloading HHS Hospitalization data")
    # hosp_url = "https://healthdata.gov/node/3565481/download"
    hosp_url = "https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD"

    filename = bucky_cfg["data_dir"] + "/cases/hhs_hosps.csv"

    # Download case and death data
    context = ssl._create_unverified_context()  # pylint: disable=W0212  # nosec
    # Create filename
    with urllib.request.urlopen(hosp_url, context=context) as testfile, open(  # nosec
        filename,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(testfile.read().decode())

    # Map state abbreviation to ADM1
    hhs_data = pd.read_csv(filename)
    abbrev_to_adm1_code = pd.read_csv(bucky_cfg["data_dir"] + "/us_adm1_abbrev_map.csv")
    abbrev_to_adm1_code = abbrev_to_adm1_code.set_index("state")
    abbrev_map = abbrev_to_adm1_code.to_dict()["adm1"]
    hhs_data["adm1"] = hhs_data["state"].map(abbrev_map)
    hhs_data.to_csv(filename)


def main():
    """Uses git to update public data repos."""
    # Repos to update
    repos = [
        bucky_cfg["data_dir"] + "/cases/COVID-19/",
        bucky_cfg["data_dir"] + "/mobility/DL-COVID-19/",
        bucky_cfg["data_dir"] + "/mobility/COVIDExposureIndices/",
        bucky_cfg["data_dir"] + "/vac/covid19-vaccine-timeseries/",
        bucky_cfg["data_dir"] + "/vac/county-acip-demos/",
    ]

    for repo in repos:
        git_pull(repo)

    # Process CSSE data
    process_csse_data()

    # Process COVID Tracking Data
    # update_covid_tracking_data()

    # Process USA Facts
    # update_usafacts_data()

    # Get HHS hospitalization data
    update_hhs_hosp_data()


def git_pull(abs_path):
    """Updates a git repository given its path.

    Parameters
    ----------
    abs_path : str
        Abs path location of repository to update
    """
    git_command = "git pull --rebase origin master"

    # pull
    with subprocess.Popen(git_command.split(), stdout=subprocess.PIPE, cwd=abs_path) as p:
        output, error = p.communicate()

    if error:

        # Get name of repo
        # git_name = "git remote -v"
        # process = subprocess.Popen(git_name.split(), stdout=subprocess.PIPE)
        logging.error("Error pulling from repo: " + output)


if __name__ == "__main__":
    main()
