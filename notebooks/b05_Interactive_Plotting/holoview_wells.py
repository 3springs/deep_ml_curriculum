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
#     display_name: deep_ml_curriculum
#     language: python
#     name: deep_ml_curriculum
# ---

# +
import numpy as np
import pandas as pd
import holoviews as hv
from bokeh import palettes

hv.extension("matplotlib")
# -

import pandas as pd
import xarray as xr
xf = xr.open_zarr("../../data/processed/geolink_norge_dataset/geolink_norge_well_logs.zarr")
xf = xf.where(xf['Well'].str.startswith('30')).dropna(dim='Well', how='all')
df = xf.to_dataframe(dim_order=['Well', 'DEPT'])
df['LITHOLOGY_GEOLINK'] = df['LITHOLOGY_GEOLINK'].astype('category')
df['Well'] = df.index.get_level_values(0).astype('category')
df['DEPT'] = df.index.get_level_values(1)
feature_cols = ['CALI', 'DTC', 'GR', 'RDEP', 'RHOB',
       'RMED', 'xc', 'yc', 'DEPT']
df = df.dropna(how='any', subset=feature_cols+['LITHOLOGY_GEOLINK'])
df = df.sort_index()
df

# +


# custom qualitative colormap
color_dict = {'Aeolian Sandstone': '#ffffe0',
 'Anhydrite': '#ff80ff',
 'Argillaceous Limestone': '#1e90ff',
 'Arkose': '#eedd82',
 'Basement': '#fa8072',
 'Biogenic Ooze': '#CCCC00',
 'Calcareous Cement': '#00ffff',
 'Calcareous Debris Flow': '#40e0d0',
 'Calcareous Shale': '#008b8b',
 'Carnallite': '#ff00ff',
 'Chalk': '#6a5acd',
 'Cinerite': '#00ffff',
 'Coal': '#000000',
 'Conglomerate': '#ffffe0',
 'Cross Bedded Sst': '#ffd700',
 'Dolomite': '#00ffff',
 'Gap': '#ffffff',
 'Halite': '#ffc0cb',
 'Kaïnite': '#fff0f5',
 'Limestone': '#6a5acd',
 'Marlstone': '#00bfff',
 'Metamorphic Rock': '#008b8b',
 'Plutonic Rock': '#ff0000',
 'Polyhalite': '#ffb6c1',
 'Porous Limestone': '#6a5acd',
 'Sandstone': '#ffff00',
 'Sandy Silt': '#d2b48c',
 'Shale': '#008b8b',
 'Shaly Silt': '#CCCC00',
 'Silt': '#ffa07a',
 'Silty Sand': '#ffffe0',
 'Silty Shale': '#006400',
 'Spiculite': '#939799',
 'Sylvinite': '#ff80ff',
 'Volcanic Rock': '#ffa500',
 'Volcanic Tuff': '#ff6347',
}


# -

well_name='30_4-1'
logs = df.loc[well_name].loc[0:3500]
facies=logs['LITHOLOGY_GEOLINK'].astype('category').values
cmap = [color_dict.get(n, 'white') for n in facies.categories][1:]
facies_image=np.repeat(np.expand_dims(facies.codes,1), 100, 1)

hv.Image(facies_image, bounds=(0, 0, 100, 100))



# +
from matplotlib import pyplot as plt

def plot_well_facies(well_name='30_4-1', depth_start=0, depth_end=6000):
    logs=df.loc[well_name]
    if depth_start is not None:
        logs = logs.loc[depth_start:depth_end]
    depth_start=logs['Depth'].min()
    depth_end=logs['Depth'].max()
    facies=logs['LITHOLOGY_GEOLINK'].astype('category').values
    cmap = [color_dict.get(n, 'white') for n in facies.categories][1:]
    facies_image=np.repeat(np.expand_dims(facies.codes,1), 500, 1)
    formatter = plt.FuncFormatter(lambda val, loc: facies.categories[val])
    img = hv.Image(facies_image, bounds=(0, -depth_end, 100, -depth_start)).opts(
        title=well_name, 
        cmap=cmap, 
        colorbar=True, 
        cformatter=lambda x:facies.categories[x],
        clabel='facies',
        xaxis=None)
    return img

dmap = hv.DynamicMap(plot_well_facies, kdims=["well_name", "depth_start", "depth_end"])
dmap = dmap.redim.range(depth_start=(0, 6000-1), depth_end=(1, 6000))
dmap = dmap.redim.values(well_name=list(df.Well.unique()))
dmap = dmap.redim.default(well_name="30_4-1", depth_start=0, depth_end=6000)
dmap
# -



def cformatter(x):
    return facies.categories[x]
cformatter = plt.FuncFormatter(lambda val, loc: facies.categories[val])
hv.Image(facies_image, bounds=(0, -6000, 600, 60)).opts(title=well_name, 
                                                cmap=cmap, 
                                                colorbar=True, 
                                                xaxis=None,
                                                cformatter=cformatter, 
                                                clabel='test'
                                               )

img = hv.Image(facies_image).opts(cmap=cmap, colorbar=True, cformatter=cformatter)
print(img)
img

# +
# hv.help(hv.Image)
# -

from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
fixed_scale_grid = rasterize(img).opts(
#     title='Fixed color range', 
    clim=(-0.5, len(facies.categories)-0.5), 
    cmap=cmap)
fixed_scale_grid








