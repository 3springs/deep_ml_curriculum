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
#     display_name: deep_ml_curriculum
#     language: python
#     name: deep_ml_curriculum
# ---

# +


import torch
from torch import nn, optim
from torch import functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import xarray as xr

# + [markdown] colab_type="text" id="kzlqXAj4EIBN"
# In this example we are going to look at well logs which are sequential data as well.

# + colab={"base_uri": "https://localhost:8080/", "height": 255} colab_type="code" id="uNl846nE-jjq" outputId="de7b4197-6a3f-4e88-e07e-2463adba90d0"
import pandas as pd
import xarray as xr
xf = xr.open_zarr("../../data/processed/geolink_norge_dataset/geolink_norge_well_logs.zarr")
xf = xf.where(xf['Well'].str.startswith('30')).dropna(dim='Well', how='all')
df = xf.to_dataframe().swaplevel()
df['LITHOLOGY_GEOLINK'] = df['LITHOLOGY_GEOLINK'].astype('category')
df['Well'] = df.index.get_level_values(0).astype('category')
df['DEPT'] = df.index.get_level_values(1)
feature_cols = ['CALI', 'DTC', 'GR', 'RDEP', 'RHOB',
       'RMED', 'xc', 'yc', 'DEPT']
df = df.dropna(how='any', subset=feature_cols+['LITHOLOGY_GEOLINK'])
df = df.sort_index()
df
# -

# %reload_ext autoreload
# %autoreload 2

# +
# DEBUG plot
from deep_ml_curriculum.visualization.well_log import plot_facies, plot_well, plot_well_pred
well_name = "30_4-1"
a=5200
b=5380
logs = df.loc[well_name].loc[a:b].copy()
facies = logs['LITHOLOGY_GEOLINK'].astype('category').values

facies = facies.add_categories('[Hidden]')
facies_true = facies.copy()
facies[-100:] = '[Hidden]'

f, ax= plot_well_pred(
    well_name, 
    logs, 
    facies_true=facies_true,
    facies_pred=facies,
)
ax[-2].set_xlabel("Facies\n(model input)")
plt.savefig(dpi=300, fname='plot.png')

# +
# plt.yticks?
# -

# context length of 150.0 m or 1000 intervals
# model can see human labels up to 15.0m above. Or 100 intervals
