from pathlib import Path
import pandas as pd
from datetime import datetime
import us
from epiweeks import Week


def parse_epiweeks(df):
    """ Converts the week and year information into a datetime stamp. """

    dates = []
    for _, row in df.iterrows():
        week_num = int(row['week'])
        year = int(row['year'])
        dates.append(Week(year, week_num).enddate())
    df = df.assign(date= dates)

    return df

def parse_states(df):
    """ Converts the region into state abbreviation. """

    state_abbrs = []
    for _, row in df.iterrows():
        region = row['region'].lower()
        if region.startswith('commonwealth'):
            region = 'Northern Mariana Islands'
        elif region == 'new york city':
            region = 'new york'
        state_abbrs.append(us.states.lookup(region).abbr)

    df = df.assign(state = state_abbrs)
    df.drop(columns=['region'], inplace=True)

    return df

def transform(output_file):
    """ Pulls data from the HHS Protect System at HealthData.gov of influenza deaths.

        Arguments:
         - output_file [str]: file name for processed .csv file
    """

    base_dir = Path().cwd()
    hhs_file = base_dir / "hhs_hosp.csv"
    hhs_df = pd.read_csv(hhs_file)

    ilinet_file = base_dir / "ILINet.csv"
    ili_df = pd.read_csv(ilinet_file).reset_index()

    # process ILINet data
    col_names = [i.lower() for i in  ili_df.iloc[0]]
    ili_df  = pd.DataFrame(ili_df.values[1:], columns=col_names)
    ili_df = parse_epiweeks(ili_df)
    ili_df = parse_states(ili_df)

    # convert the date for merging
    hhs_df['date'] = [datetime.strptime(str(date), '%Y/%m/%d') for date in hhs_df['date']]
    hhs_df['date'] = hhs_df['date'].astype(str)
    ili_df['date'] = ili_df['date'].astype(str)

    # left merge on the incidences (ILINet)
    df = ili_df.merge(hhs_df, on=['date','state'], how='left')

    cumul_report_cases = 'ilitotal'
    cumul_deaths = 'previous_day_deaths_influenza'

    df = df[['state', 'date', cumul_report_cases, cumul_deaths]]

    # clean up missing values 
    df = df.fillna(0)
    df['previous_day_deaths_influenza'] = df['previous_day_deaths_influenza'].cumsum()

    # convert adm1 to adm2 by affixing '001' to the adm1 code
    # TODO: find a better method of mapping adm1 to adm2
    state_abbr_map = us.states.mapping("abbr", "fips")
    adm2_abbr = {}
    for state, fips in state_abbr_map.items():
        if fips is None:
            adm2_abbr[state] = None
        else:
            adm2_abbr[state] = str(fips) + '001'

    df['adm2'] = df['state'].map(adm2_abbr)

    # convert region code to int type
    df['adm2'] = pd.to_numeric(df['adm2'])

    # some processing
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={cumul_report_cases: 'cumulative_reported_cases',
                       cumul_deaths: 'cumulative_deaths'}, inplace=True)

    df = df.set_index(['adm2', 'date']).sort_index()

    # NOTE: removing rows corresponding to mappings of fips 66 and 69
    if 66001 in df.index.unique('adm2'):
        df.drop([66001], inplace=True)
    if 69001 in df.index.unique('adm2'):
        df.drop([69001], inplace=True)

    df.to_csv(output_file, index=True)
