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

# # Plotting libraries
#
# We mention a few plotting libraries:
#
# - [Matplotlib](https://matplotlib.org/3.1.1/gallery/index.html) is for basic plotting, it is the classic library that can do 80% of your plots
# - [Seaborn](https://seaborn.pydata.org/examples/index.html) is for statistical visualization 
# - [Holoviews](http://holoviews.org/gallery/index.html) - very high-level tool for interactive dashboards and large data (when using datashader backend). Not as well-suited to making figures for publication.
#     - Bokeh is usefull for interactive plotting but because it's browser based, it can struggle with large amounts of data, unless you use server-based plotting. We will use it as a Holoviews backend but it's worth looking into by itself.
#     - Datashader is useul for plotting large quantity of data points. We will use it as a holoview backend
#
# But the important thing is to get get good at one so you can easily produce plots. Matplotlib is the default choice unless you expect to need interactivity or large datasets.
#
# Most plotting workflows start by finding an example and modifying it. For that we browse the galleries.

# # Datashader
# Datashader is a library that extends Bokeh and it is useul for plotting large quantity of data points. Using Datashader is highly recommended when dealing with more than a million data points, especially when other plotting libraries have trouble visualising the data. Datashader uses aggregation to visualise data, which means we are not seeing every single data point rather an aggregated form of the data. The output of datashader is an image, which can be used by other libraries. Holoviews is one of the packages which is based on bokeh and take advantage of Datashader for ploting large datasets.<br>
# Datashader only supports a few types of plots:
# - Scatter plots and heatmaps
# - Trajectories
# - Rasters

import datashader as ds

# +
import pandas as pd
import numpy as np

np.random.seed(14)
n = 500000
df1 = pd.DataFrame(
    {
        "x": np.random.normal(0, 2, n),
        "y": np.random.normal(0, 2, n),
        "z": np.random.normal(1, 0.1, n),
        "cat": "A",
    }
)
df2 = pd.DataFrame(
    {
        "x": np.random.normal(3, 1, n),
        "y": np.random.normal(0, 1, n),
        "z": np.random.normal(5, 0.2, n),
        "cat": "B",
    }
)
df3 = pd.DataFrame(
    {
        "x": np.random.normal(-3, 1, n),
        "y": np.random.normal(2, 1, n),
        "z": np.random.normal(10, 0.5, n),
        "cat": "C",
    }
)

df = pd.concat([df1, df2, df3], ignore_index=True)
df["cat"] = df["cat"].astype("category")
df.head()

# +
import datashader as ds
import datashader.transfer_functions as tf

tf.shade(ds.Canvas().points(df, "x", "y"))
# -

# ## Datashader pipeline
# Datashader follows these steps to generate an image from data:
# 1. Projection
# 2. Aggregation
# 3. Transformation
# 4. Colormapping
# 5. Embedding
#
# Let's see how each step is done.

# ### Projection and Aggregation
# Here, projection means choosing the features in the data we want to visualise. In a scatter plot we are going to have an x-axis and a y-axis, so we want to decide which column is going to be _x_ and which column is going to be _y_. Since datashader visualises an aggregated form of the data in an image, we also need to decide what aggregation function should be used. Datashader supports the following functions:
# - `any()`: returns __*1*__ if any data point exists in the pixel, otherwise returns __*0*__.
# - `count()`: returns the number of data points in each pixel.
# - `count_cat()`: returns the number of data points for each category in each pixel.
# - `sum()`: the total value of all the numbers in a pixel (we can also specify which column should be used).
#
#

# First, we need to create an empty canvas.

canvas = ds.Canvas(plot_width=400, plot_height=400, x_range=(-5, 5), y_range=(-5, 5))

# Then we pass in the data, the columns we want to use , and the aggregation function.

aggr = canvas.points(df, "x", "y", agg=ds.count())
aggr

tf.shade(aggr)

# ### Transformation
# In this step we apply any form of mathematical function if necessary. For instance, we may want to have only the top 10% values in the plot, or apply a `log` function to data.

tf.shade(np.log(aggr + 1))

# ### Colormapping
# Here we can select the colormap of the visualisation using Bokeh palettes.

tf.shade(aggr, cmap=["blue", "red"])

# When we are dealing with categorical variables we can assign a color to each category.

colors = {"A": "red", "B": "blue", "C": "green"}
aggr = canvas.points(df, "x", "y", agg=ds.count_cat("cat"))
tf.shade(aggr, colors)

# We can make the points appear larger by spreading them.

tf.spread(tf.shade(aggr, colors))

# ## With HoloViews
# The image generated by datashader can be used like a normal image, or can be visualised via Holoviews. Using holoviews allows us to quickly render the image when we zoom in and out.

# Holoviews has a range of datashader operations available
from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from holoviews.operation import decimate



# - rasterize: uses an aggregator and interpolation method to turn data into an image
# - shade: normalise and colormaps to an image
# - datashade: combines the two main computations done by datashader, namely shade() and rasterize():
# - dynspread: dynamically makes isolated pixels "spread" into adjacent ones for visibility.
#
# The primary one to use is datashade

# +
import holoviews as hv

hv.extension("bokeh")

colors = {"A": "red", "B": "blue", "C": "green"}
pts = hv.Points(df, ["x", "y"])
plot = datashade(pts, color_key=colors, aggregator=ds.count_cat("cat"))
# don't show pts, it will freeze your browser
# plot
# -

# By itself this changes the color
rasterize(pts)

# rasterize + shade = datashade
shade(rasterize(pts)) + datashade(pts)

# Spreading makes it easier to see when we zoom in on isolated points.

datashade(pts) + dynspread(datashade(pts))

# <div class="alert alert-success">
# <h2>Exercise</h2>
#
# Use Holoviews with Datashader to view our smart meter data.
#     
# Use the code below replacing the `?` and entering your code where marked
#
#
# ```python
# import pandas as pd
# import datashader as ds
# import datashader.transfer_functions as tf
#
# # Load data
# df = block0 = pd.read_csv("../../data/processed/smartmeter/block_0.csv", parse_dates=['day'], index_col=['day'])[['LCLid', 'energy_sum']]
# print(df.head())
#
# # Make it into points
# pts = hv.Points(
#     # YOUR CODE HERE
#     ).opts(color='?')
# colors = ['red', 'green', 'blue']*30 # We need 50+ colors, here's a quick way
# plot = datashade(pts, color_key=colors, aggregator=ds.count_cat("?"))
# plot
# ```
#
#
# <details>
# <summary><b>→ Hints</b></summary>
#
# * View day as the x axis, energy_sum as the y axis
# * Aggregate and color by LCLid
#
# </details>
#
# <br/>
# <br/>
# <details>
# <summary>
# <b>→ Solution</b>
# </summary>
#
# ```python
# import pandas as pd
# import datashader as ds
# import datashader.transfer_functions as tf
#
# # Load data
# df = block0 = pd.read_csv("../../data/processed/smartmeter/block_0.csv", parse_dates=['day'], index_col=['day'])[['LCLid', 'energy_sum']]
# print(df.head())
#
# # Make it into points
# pts = hv.Points(df, ["day", "energy_sum"], vdims=['LCLid']).opts(color='LCLid')
# colors = ['red', 'green', 'blue']*30
# plot = datashade(pts, color_key=colors, aggregator=ds.count_cat("LCLid"))
# plot
# ```
#
# </details>
#
# </div>

# For more information on using datashader with holoviews see:
# - http://holoviews.org/user_guide/Large_Data.html
#
# You may also consider Bokeh server: https://docs.bokeh.org/en/latest/docs/user_guide/server.html




