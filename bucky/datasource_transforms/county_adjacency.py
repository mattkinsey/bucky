from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def calculate(output_filename, census_data_path):

    base_dir = Path().cwd()

    # Get county populations
    census_df = pd.read_csv(census_data_path, index_col="adm2")
    county_pop = census_df.sum(axis=1).rename("population")

    # Read shapefile
    county_df = gpd.read_file(base_dir / "tl_2021_us_county.shp")
    county_df["adm2"] = county_df["GEOID"].astype(int)
    county_df = county_df.set_index("adm2").sort_index()
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

    output_df.to_csv(output_filename, index=True)
