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

# Data from https://drive.google.com/drive/u/0/folders/1u6cxK5hekpnQw9cvd2khp-MDPZyoUxib
#
# Work in progress

import lasio
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from tqdm.auto import tqdm
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path

data_in = Path('/media/wassname/Storage5/ml_project_data/deep-ml/All wells data logs and reports/torosa 1/Mini_DSTand_MDT/pdplot_well_1/test_point_time_index_data')

data=[]
for f in data_in.glob('*.las'):
    df = lasio.read(f).df()
    df['well']=f.stem
    data.append(df)
#     break
data2 = pd.concat(data)
data2

df.columns

# show full curves dict
files = sorted(data_in.glob('*.las'))
f = next(iter(files))
log = lasio.read(f)
print(f.stem)
log.curvesdict

l = lasio.read(f)
l

dir(l)

pressure = [c['mnemonic'] for c in l.curves if 'PSI' in c['unit']]

for df in data:
    df = df.dropna(thresh=0.5)
    df = df[(df.std()>1).index]
    df -= df.mean()
    df /= df.std()
    df[pressure].plot()
#     plt.legend('off')
    plt.show()

names={name:curve for name,curve in l.curvesdict.items()}
for df in data:
    for n,c in names.items():
        if c['unit'] == 'PSI':
            print(n,c)
            df[n].plot()
    plt.show()

for df in data:
    df['POUDHP'].plot()

a,b=list(l.curvesdict.items())[10]
names={name:curve['descr'] for name,curve in l.curvesdict.items()}
df = l.df()
print(names)
df.rename(columns=names).columns
# names.

l.df()


