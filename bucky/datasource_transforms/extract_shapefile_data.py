from multiprocessing import Pool, cpu_count
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import us
from scipy.spatial.distance import pdist, squareform


def extract(output_connectivity_filename, output_adm_filename, census_data_path):

    base_dir = Path().cwd()

    # Get county populations
    census_df = pd.read_csv(census_data_path, index_col="adm2")
    county_pop = census_df.sum(axis=1).rename("population")

    # Read shapefile
    county_df = gpd.read_file(base_dir / "tl_2021_us_county.shp")

    # Get adm codes
    county_df["adm2"] = county_df["GEOID"].astype(int)
    county_df["adm1"] = county_df["STATEFP"].astype(int)
    county_df["adm0"] = 0

    # Get adm names and abbrs
    county_df["adm2_name"] = county_df["NAMELSAD"]
    county_df["adm2_abbr"] = county_df["NAME"]

    fips_name_map = us.states.mapping("fips", "name")
    fips_abbr_map = us.states.mapping("fips", "abbr")

    county_df["adm1_name"] = county_df["STATEFP"].map(fips_name_map)
    county_df["adm1_abbr"] = county_df["STATEFP"].map(fips_abbr_map)

    county_df["adm0_name"] = "United States"
    county_df["adm0_abbr"] = "US"

    # Sort by adm2
    county_df = county_df.set_index("adm2").sort_index()

    # add total population
    county_df = pd.merge(county_df, county_pop, left_index=True, right_index=True, how="left")

    # save adm mapping
    county_df.to_csv(
        output_adm_filename,
        index=True,
        columns=[
            "adm1",
            "adm0",
            "adm2_name",
            "adm2_abbr",
            "adm1_name",
            "adm1_abbr",
            "adm0_name",
            "adm0_abbr",
            "population",
        ],
    )

    # project the polygons so we can calculate distances
    county_df = county_df.to_crs("EPSG:3087")  # https://epsg.io/3087

    county_centroids = county_df.centroid

    county_x = county_centroids.apply(lambda x: x.x).to_numpy()
    county_y = county_centroids.apply(lambda x: x.y).to_numpy()

    county_coords = np.stack([county_x, county_y]).T

    condensed_dists = pdist(county_coords)
    sq_dists = squareform(condensed_dists)

    output_df = pd.DataFrame(sq_dists, index=county_df.index.rename("j"), columns=county_df.index.rename("i")).unstack()
    output_df = pd.DataFrame(output_df, columns=["distance"])
    output_df = pd.merge(output_df, county_pop, left_on="i", right_index=True, how="left")
    output_df = pd.merge(output_df, county_pop, left_on="j", right_index=True, how="left")
    output_df = output_df.rename(columns={"population_x": "i_pop", "population_y": "j_pop"})

    n_cpu = min(cpu_count(), 4)
    with Pool(n_cpu) as pool:
        result = pool.map(county_df.geometry.touches, county_df.geometry.values)

    touches_mat = np.stack([res.values.astype(int) for res in result])
    output_df["touches"] = touches_mat.reshape(-1)

    output_df.to_csv(output_connectivity_filename, index=True)
