# -*- coding: utf-8 -*-
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

# From https://github.com/Philliec459/Carbonates--Generate-Representative-Thin-Sections-and-Pc-Curves-from-normalized-core-data-using-KNN
#
# 1 Clerke, E. A., Mueller III, H. W., Phillips, E. C., Eyvazzadeh, R. Y., Jones, D. H., Ramamoorthy, R., Srivastava, A., (2008) “Application of Thomeer Hyperbolas to decode the pore systems, facies and reservoir properties of the Upper Jurassic Arab D Limestone, Ghawar field, Saudi Arabia: A Rosetta Stone approach”, GeoArabia, Vol. 13, No. 4, p. 113-160, October, 2008.

import pandas as pd
from pathlib import Path
import PIL
from IPython.display import display
# %pylab inline

datadir_in = Path("../../data/raw/thin-sections-and-pc-curves-from-core-data/")

df = pd.read_excel(datadir_in / "CO3_TS_Image.xls", header=None)
df.columns = ["id", "Depth", "Porosity", "Permeability", "Path"]
df

X = []
for i in range(len(df)):
    p = df.Path[i]
    im = PIL.Image.open(datadir_in / p)
    x = np.array(im.resize((256, 256)))
    X.append(x)
    if i < 5:
        display(im)
        print(x.shape, im.size, p, df.iloc[i])
X = np.stack(X)
X.shape






