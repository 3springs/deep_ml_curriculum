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

# # Context
#
# The crude oil price movements are subject to diverse influencing factors. This dataset was retrieved from the U.S. Energy Information Administration: Europe Brent Spot Price FOB (Dollars per Barrel)
# Content
#
# The aim of this dataset and work is to predict future Crude Oil Prices based on the historical data available in the dataset.
# The data contains daily Brent oil prices from 17th of May 1987 until the 30th of September 2019.
# Acknowledgements
#
# Dataset is available on U.S. Energy Information Administration: Europe Brent Spot Price FOB (Dollars per Barrel) which is updated on weekly bases.
#
# from https://www.eia.gov/dnav/pet/hist/rbrteD.htm

import lasio
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from tqdm.auto import tqdm
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

data_in = Path("../../data/raw/brent_oil_price/")
data_out = Path("../../data/processed/brent_oil_price")

# # Load, process

df = pd.read_excel(data_in / "RBRTEd.xls", sheet_name="Data 1", header=2)
df.Date = pd.to_datetime(df.Date)
df = df.set_index("Date")
df = df.rename(columns={"Europe Brent Spot Price FOB (Dollars per Barrel)": "RBRTEd"})
df

data_out.mkdir(parents=True, exist_ok=True)
df.to_parquet(data_out / "RBRTEd.parquet", compression="gzip")

df.plot()


