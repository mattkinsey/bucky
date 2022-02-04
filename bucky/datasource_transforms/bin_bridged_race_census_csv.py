import io

import numpy as np
import pandas as pd


def _fixed_width_to_csv(file, blocks):
    slices = [slice(*b) for b in blocks]
    with open(file, "r") as f:
        data = f.readlines()

    fixed_lines = [",".join([line[s].strip() for s in slices]) for line in data]
    return io.StringIO("\n".join(fixed_lines))


def bin_census_data(filename, out_filename):
    """Group ages in the Census csv to match the bins used by Prem et al.
    Parameters
    ----------
    filename : str, path object or file-like object
        Unmodified Census CSV
    out_filename : str, path object or file-like object
        Output file for binned data
    """
    # Preprocess the file and add delimiters because idk who invented this fixed width format...
    # see pg 17 of https://www.cdc.gov/nchs/data/nvss/bridged_race/Documentation-Bridged-PostcenV2020.pdf
    # if you want your brain to melt

    blocks = [(4, 9), (9, 11), (101, 109)]
    csv_f_obj = _fixed_width_to_csv(filename, blocks)

    df = pd.read_csv(csv_f_obj, names=["adm2", "age", "N"], dtype=int)
    df["age_group"] = pd.cut(df["age"], np.append(np.arange(0, 76, 5), 120), right=False)

    df = df.groupby(["adm2", "age_group"]).sum()[["N"]].squeeze().unstack("age_group")

    # add in territory data (included with bucky)
    territory_df = pd.read_csv("../included_data/population/territory_pop.csv", index_col="adm2")
    territory_df.columns = df.columns

    df = pd.concat([df, territory_df])

    df.to_csv(out_filename)
