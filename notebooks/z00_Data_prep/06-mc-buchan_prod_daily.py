# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: jup3.7.3
#     language: python
#     name: jup3.7.3
# ---

# From https://data-ogauthority.opendata.arcgis.com/datasets/daily-buchan-production-data
#

import lasio
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from tqdm.auto import tqdm
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

data_in = Path("../../data/raw/daily_buchan_prod_data/")
data_out = Path("../../data/processed/daily_buchan_prod_data")

# # Load, process





df = pd.read_csv(data_in / "Daily_Buchan_Production_Data.csv")
df.index = pd.to_datetime(df["DATE_"])
df = df[df["PRODUCED_GAS_GAS_MMCF"] > 0]
df = df.drop(columns=["DATE_", "OBJECTID"])

gdf2 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs="EPSG:4230")
data_out.mkdir(parents=True, exist_ok=True)
gdf2.to_file(data_out / "Daily_Buchan_Production_Data.gpkg", driver="GPKG")
gdf2



# ## Load

gdf = gpd.read_file(data_out / "Daily_Buchan_Production_Data.gpkg")
gdf.index = pd.to_datetime(gdf["DATE_"])
gdf

for name, df in gdf.groupby("WELLID"):
    df["DAILY_WELL_DRY_OIL_BBL"].plot(label=name)
plt.title("DAILY_WELL_DRY_OIL_BBL")
plt.legend()
plt.ylabel("DAILY_WELL_DRY_OIL_BBL")
plt.yscale("log")

# +
# aggregate over wells and fields
df_field = (
    gdf.dropna()
    .resample("1D")
    .sum()[["DAILY_WELL_DRY_OIL_BBL", "PRODUCED_GAS_GAS_MMCF", "PRODUCED_WATER_BBL"]]
)  # .dropna()

df_field["DAILY_WELL_DRY_OIL_BBL"].plot()
df_field
# -






