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

# Data from http://www.pressureplot.com/data.aspx
#     
# See the specific liscence for terms
#
# this notebook is a work in progress

import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import PIL
from IPython.display import display
# %pylab inline

datadir_in = Path("../../data/raw/PHGDatabase/PHGDatabase_April2019.mdb")
datadir_out = Path("../../data/processed/PHGDatabase/PHGDatabase_April2019")
datadir_in.exists()
datadir_out.mkdir(exist_ok=True, parents=True)

# # export from mdb



# +
cmd = 'mdb-tables {}'.format(datadir_in)
tables = getoutput(cmd).split()
tables

for table in tables:
    cmd = 'mdb-export {} {}'.format(datadir_in, table) % ()
    csv = getoutput(cmd)
    fo = '{}/{}.csv'.format(datadir_out, table)
    print(fo)
    open(fo, "w").write(csv)

# +
#
# -

# # import from csv

tables = {}
fs = sorted(datadir_out.glob("*.csv"))
for f in tqdm(fs):
    df = pd.read_csv(f)
    df.name = f.stem
    tables[f.stem] = df

# list tables
pd.Series({k: len(v) for k, v in tables.items()}).sort_values()





df_dst = tables["PHG_Dst_Horner"].copy()
df_dst["WELL_NAME"] = df_dst["WELL_ID"].replace(replace_id2name)
df_dst = df_dst.dropna()
df_dst

# need gradient for each test
for i, (n, g) in enumerate(df_dst.groupby("WELL_NAME")):
    ax = plt.gca()
    ax2 = plt.twiny()
    g.plot.scatter(x="PRESSURE", y="DELTA_T", ax=ax, c="blue")
    g.plot.scatter(x="TEMPERATURE", y="DELTA_T", ax=ax2, c="y")
    plt.title('{}'.format(n))
    plt.show()
    if i > 2:
        break

# +
# import matplotlib.cm as cm

# x = np.arange(15)
# ys = [i+x+(i*x)**2 for i in range(15)]

# colors = cm.rainbow(np.linspace(0, 1, len(ys)))

# +
replace_id2name = tables["PHG_Well"].set_index("WELL_ID")["WELL_NAME"]
replace_lith = tables["PHG_LithType"].set_index("LITHOLOGY_TYPE_CODE")[
    "LITHOLOGY_TYPE_DESCRIPTION"
]

# get poro perm
df = tables["PHG_Porperm_View"].copy()
df["WELL_NAME"] = df["WELL_ID"].replace(replace_id2name).values
df["Lithology"] = df["LITHOLOGY_CODE"].replace(replace_lith).values


df_poroperm = df[
    [
        "WELL_NAME",
        "TOP_INTERVAL",
        "PERMEABILITY",
        "POROSITY",
        "Lithology",
        "GRAIN_DENSITY",
    ]
].set_index(["WELL_NAME", "TOP_INTERVAL"])
df_poroperm


for i, (n, g) in enumerate(df_poroperm.groupby("WELL_NAME")):
    if i > 15:
        break
    if len(g) > 10:
        im = plt.scatter(x=g["PERMEABILITY"], y=g["POROSITY"], c=g.xs(n).index)
        plt.title('well name: {}'.format(n))
        plt.colorbar(label="TOP_INTERVAL")
        plt.show()
# -






