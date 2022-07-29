from pathlib import Path
import pandas as pd
import numpy as np
import us


def transform(output_file):
    """ Pulls data from the HHS Protect System at HealthData.gov of influenza hospitalizations.

        Arguments:
         - output_file [str]: file name for processed .csv file
    """

    base_dir = Path().cwd()
    hhs_file = base_dir / "hhs_hosp.csv"
    df = pd.read_csv(hhs_file)
    
    # define target feats
    inc_hosps = 'previous_day_admission_influenza_confirmed'
    current_hosps = 'total_patients_hospitalized_confirmed_influenza'

    df = df[['state', 'date', inc_hosps, current_hosps]]

    # map state to fips code
    state_abbr_map = us.states.mapping("abbr", "fips")
    df['adm1'] = df['state'].map(state_abbr_map)

    # convert region code to int type
    df['adm1'] = pd.to_numeric(df['adm1'])

    # some processing
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={inc_hosps: 'incident_hospitalizations', current_hosps: 'current_hospitalizations'},
                       inplace=True)

    df = df.set_index(['adm1', 'date']).sort_index()


    # Add in nan rows for the missing territories
    if 66 in df.index.unique('adm1'):
        raise RuntimeError('Unexpected adm1 66 in HHS Data')
    if 69 in df.index.unique('adm1'):
        raise RuntimeError('Unexpected adm1 69 in HHS Data')
    empty_row_inds = pd.MultiIndex.from_product([[66, 69], df.index.unique('date')])
    df = df.reindex(df.index.append(empty_row_inds)).sort_index()

    df = df.fillna(0)

    df.to_csv(output_file, index=True)
