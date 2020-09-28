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

# Data from NOPIMS, see readme in data dir

import lasio
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from tqdm.auto import tqdm
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

data_in = Path("../../data/raw/cleo3_DST/P00687700/clio_3_dst_gauge_data/")
data_out = Path("../../data/processed/cleo3_DST/P00687700/clio_3_dst_gauge_data")


# +
def read_tpr(f, header=14, widths=[10 + 12, 12, 12, 9]):
    df = pd.read_fwf(f, header=header, widths=widths)
    with f.open() as fi:
        header = "".join([fi.readline() for i in range(header)])

    return df, header


# extra parsing for this clio3 PSIA files
def parse_clio3_tpr(df):
    df["date"] = pd.to_datetime(df["Date    Hr Min Sec"])
    df_psia = df.set_index("date")[["PSIA"]]
    return df_psia


# -

f = data_in / "Oculus Tailpipe Gauge 23403 Set Below Seal Assembly-head.TPR"
df, header = read_tpr(f, header=14, widths=[10 + 12, 12, 12, 9])
df_psia = parse_clio3_tpr(df)
df_psia.plot()
df_psia

f = data_in / "Oculus Tailpipe Gauge 23403 Set Below Seal Assembly.TPR"
df, header = read_tpr(f, header=14, widths=[10 + 12, 12, 12, 9])
df_psia = parse_clio3_tpr(df)
df_psia.plot()
df_psia



# # save as parquet

fo = data_out / (f.stem.replace(" ", "_") + "PSIA.parquet")
fo.parent.mkdir(parents=True, exist_ok=True)
print("save to", fo)
df_psia.to_parquet(fo, compression="gzip")

df_psia = pd.read_parquet(fo)

# lots of missing intervals
df_psia_r = df_psia.resample("1T").mean()
df_psia_r.plot()

# show all intervals with data in
for a in pd.date_range("2009", "2011", freq="7D"):
    d = df_psia_r[a : a + pd.Timedelta("7D")]
    d = d.dropna().resample("1T").mean()
    if len(d.dropna()) > 10:
        d.plot()



# +
# # Work out widths
# a='31/07/10 |09:00:03   |125.90083  |6166.8823  |128.6929'
# widths=[len(aa)+1 for aa in a.split('|')]
# widths
# -




