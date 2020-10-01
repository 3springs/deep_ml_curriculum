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
# - [Holoviews](http://holoviews.org/gallery/index.html) - very high-level tool for interactive dashboards and large data (when using datashader too). Not as well-suited to making figures for publication.
#     - Bokeh is usefull for interactive plotting but because it's browser based, it can struggle with large amounts of data, unless you use server-based plotting. We will use it as a Holoviews backend but it's worth looking into by itself.
#     - Datashader is useul for plotting large quantity of data points. We will use it as a holoview backend
#
# But the important thing is to get get good at one so you can easily produce plots. Matplotlib is the default choice unless you expect to need interactivity or large datasets.
#
# Most plotting workflows start by finding an example and modifying it. For that we browse the galleries.

# # HoloViews
#
# Holoviews is a very high-level tool. It can use bokeh for plotting, making it interactive. Focused on interactive data exploration in ipython. Uses a functional, composition-based approach to plotting that is very different from everything else on the list. Extremely powerful for this task, but not as well-suited to making figures for publication.
#  
# We can optionally use datashader, which lets it handle large data.

# +
import numpy as np
import pandas as pd
import holoviews as hv
from bokeh import palettes

hv.extension("bokeh")
# -

#  <div class="alert alert-success">
#   <h2>Exercise</h2>
#
#   A lot of the work of programming is knowing what to look for and how to look. This is a bit like riding a bike - it's a habit you need to build. We sometimes have blindspots where we don't know the right jargon, or stop because we get tired or overwhelmed with unfamiliar information.
#     
#   For this excercise you need to look through the [HoloView api documentation](https://holoviews.org/Reference_Manual/index.html) and find:
#
#   1. A object let you make a line (it's not named line, but it does mention it)
#     
#   If you're unsure, please expand the hints for keywords and tips of where and how to look.
#       
#
#   <details>
#   <summary><b>→ Hints</b></summary>
#
#   * Keywords are important, they call it `curve` not line
#   * You want to go to `API` > `elements` > then find a mention of `line` or `curve`
#   * With information overload, try Cntrl-F searching first, then scanning the page
#
#   </details>
#
#   <br/>
#   <br/>
#   <details>
#   <summary>
#     <b>→ Solution</b>
#   </summary>
#
#   https://holoviews.org/Reference_Manual/holoviews.element.html#holoviews.element.Curve
#
#   </details>
#
#   </div>

# ## Creating elements
#
# All basic elements accept their data as a single, mandatory positional argument which may be supplied in a number of different formats, some of which we will now examine. A handful of *annotation* elements are exceptions to this rule, namely ``Arrow``, ``Text``, ``Bounds``, ``Box`` and ``Ellipse``, as they require additional positional arguments.
#
# ### A simple curve
#
# To start with a simple example, we will sample a quadratic function $y=100-x^2$ at 21 different values of $x$ and wrap that data in a HoloViews element:

xs = [i for i in range(-10, 11)]
ys = [100 - (x ** 2) for x in xs]
print('x', xs)
print('y', ys)

simple_curve = hv.Curve((xs, ys))
# Jupyter displays the object using its default visual representation. - a holoviews bokeh plot
simple_curve

# Here we made simple_curve, and Jupyter displays the object using its default visual representation - a Holoviews/Bokeh plot. But ``simple_curve`` itself is just a wrapper around your data, not a plot, and you can choose other representations that are not plots.  For instance, printing the object will give you a purely textual representation instead:

# Printing the same object gives us a textual representation
# a continuous mapping from `x` to `y`
print(simple_curve)

# Curve(data=None, kdims=None, vdims=None, **kwargs)
help(hv.Curve)

# What do we pass in
#
# - Element(dataframe, xdim, ydims)
# - Element(xdata, ydata)

# ### Annotating the curve
#
# There are other aspects of the data that we want to capture. For instance what are the x and y axis? Perhaps this parabola is the trajectory of a ball thrown into the air, in which case we could declare the object as:

# +
# Tell holoviews that the x axis (key dimension) is distance, the xy axis (value dimension) is height
trajectory = hv.Curve((xs, ys), kdims=["distance"], vdims=["height"])

print(trajectory)

# Holoview updates the plot
trajectory
# -

# ### Casting between elements
#

# +
# What if we tell HoloViews that the data are seperated points (a scatter) instead of connected points (a curve)
hv.Scatter(simple_curve)

# Note that we are putting a curve into a scatter and changing the type
# -

# ### Pandas - Selecting Columns
#
# Holoviews can also work with pandas data frames, which means we can pass in the name of the columns we want to see.
#
# Lets load Natural Gas production data from an [excel worksheet produced by the US Energy Information Administration](www.eia.gov/dnav/ng/ng_prod_sum_a_epg0_fgw_mmcf_m.htm)

# Here we load some synthetic natural gas production data
# ngdf_raw = pd.read_csv("../../data/processed/NaturalGas/NG_PROD_SUM.csv")
ngdf_raw = pd.read_excel(
    "../../data/processed/NaturalGas/NG_PROD_SUM.xls",
    sheet_name=2,
    header=2,
    index_col="Date",
    parse_dates=True,
)
ngdf_raw.columns = [c[: c.find(" ", 5)] for c in ngdf_raw.columns]
ngdf = ngdf_raw.resample("A").sum()
ngdf.index = ngdf.index.year
ngdf.head()



# It's a 1d series of data, so we will load it as a curve
production = hv.Curve(ngdf, kdims=["Date"], vdims=["Florida"]).opts(width=800)
production

# HoloViews recognises the name of the column as dimentions. Now, we can easily add units to the dimensions.

production = production.redim.unit(Florida="MMcf")
production

# ### Layout
# Create a layout in holoviews is very easy. First, let's try a few other types of plots to visualise the same data.

print(production)

spikes = hv.Spikes(production)
spikes

scatter = hv.Scatter(production)
scatter

area = hv.Area(production)
area

# In holoviews, creating a layout is much easier. You just simply create a layout by adding up the plots!

production + spikes + scatter + area

# To make it look better we can specify the number of columns.

layout = production + spikes + scatter + area
layout.cols(2)

print(layout)

# The object `layout` has all the plots in it. This means we can easily access each plot and even create a new layout. Note the name of the elements in the layout.

layout.Curve.I + layout.Area.I

# ### Overlays
# Another cool feature of holoviews is easy overlaying. We used `+` for layout to put plots beside each other. To put them on the top of each other we can use `*`.

layout.Curve.I * layout.Spikes.I

# And we can combine the layout and overlay.

layout.Curve.I * layout.Spikes.I + layout.Area.I

# ### Slicing and Selecting
# You can slice the elements using array-style syntax or using `.select` method. Note that in both case you need to use x-axis values to slice.

production[:2000]

production.select(Date=(2000, 2020))

production[:2000].relabel("19XX") * production.select(Date=(2000, 2020)).relabel("20XX")

# Lets try with the smart meter data
df = block0 = pd.read_csv("../../data/processed/smartmeter/block_0.csv", parse_dates=['day'], index_col=['day'])[['LCLid', 'energy_sum']]
house1 = df[df.LCLid=='MAC000002']
house2 = df[df.LCLid=='MAC005492']
df



#  <div class="alert alert-success">
#   <h2>Exercise</h2>
#
#   For house1, plot it as a curve of day vs energy_sum
#       
#
#   <details>
#   <summary><b>→ Hints</b></summary>
#
#   * Look at our pandas/production example above
#   * Intead of kvims='date', vdims='Florida' we want to use day and energy_sum
#
#   </details>
#
#   <br/>
#   <br/>
#   <details>
#   
#   <summary>
#     <i>→ Solution</i>
#   </summary>
#
#   ```python
#    hv.Curve(house1, 'day', 'energy_sum')
#   ```
#
#   </details>
#
#   </div>

#  <div class="alert alert-success">
#   <h2>Exercise</h2>
#
#   Lets pull everything together and plot the smart meter data, with one curve per household. 
#       
#
#   <details>
#   <summary><b>→ Hints</b></summary>
#
#   * `curve1 * curve2`
#
#   </details>
#
#   <br/>
#   <br/>
#   <details>
#   
#   <summary>
#     <i>→ Solution</i>
#   </summary>
#       
#       
#   ```python
#   curve1 = hv.Curve(house1, kdims=["day"], vdims=["energy_sum"])
#   curve2 = hv.Curve(house2, kdims=["day"], vdims=["energy_sum"])
#   curve1 * curve2
#       
#   ```
#
#   ```python
#     # A more complicated version, grouping by household, and using a for loop
#     curve = None
#     for i, df_household in df.groupby('LCLid'):
#
#         # It's a 1d series of data, so we will load it as a curve
#         if len(df_household):
#             curve_hh = hv.Curve(df_household, kdims=["day"], vdims=["energy_sum"])
#             curve_hh = curve_hh.redim.unit(energy_sum="kWh")
#             if curve is None:
#                 curve = curve_hh
#             else:
#                 curve *= curve_hh
#     curve
#   ```
#
#   </details>
#
#   </div>

# ## Large data with Datashader
#
# We wont go into it in depth but you can combine HoloViews with datashader to view large data. This means you can explore data with millions of points. 
#
# The quick way to do this is wrap your object in the datashader object:
#

from holoviews.operation.datashader import datashade, dynspread
dynspread(datashade(production)).opts(width=800)

# For more information on using datashader with holoviews see:
# - http://holoviews.org/user_guide/Large_Data.html
#

# ## Appearance
# So far we discussed how to visualise data. Now we will focus on the the appearance of the plot. For instance, how would we change the size of the plot? To change the appearance we need to use holoviews options system. There are three types of options in holoviews:
# - plot options: specifies how holoviews construct the plot.
# - style options specifies how the plotting extension (e.g. Bokeh) should style the plot.
# - normalisation options: specifies how holoviews normalises elements in the plot

# ### Plot Options
# Let's start by changing the size of the plot. We do this in Jupyter Notebook using magic command `%%opts`. Then we specify the type of object and the option. For instance, for changing the width of `production` plot, we use `Curves` (since it is a Curve object) and `[width=800]` to change the width to 800.

production.opts(width=800)

# %%opts Curve [width=800]
production

# To get a list of options you can use holoviews help function.

hv.help(hv.Curve)

# ### Style Options
# As mentioned before style options are to tell Bokeh (or matplotlib if we use it as the extension) how to style a plot.

production.opts(color='maroon')

# __Note:__ When we use the options, the new properties are linked to the object, which means if we wanted to plot it again we won't need to specify the color. It will keep using the same color unless we change the options.

production


# There is even a GGPlot theme, uncomment the following code to use it

# +
# # %opts Curve  [height=200 width=900 show_grid=True]
# # %opts Scatter  [height=200 width=900 show_grid=True]
# # A GGPlot theme
# from bokeh_ggplot import ggplot_theme
# hv.renderer('bokeh').theme = ggplot_theme
# production
# -

# ## Exploring data with containers
#
# When the data has multiple series it's hard see all of them. Using containers will allow us to explore the data much easier. 
#
# A Container is just a python function that creates a plot element:

def NGProduction(state="Other", year_start=1991, year_end=2020):
    """A function to make a curve (but choosing year and state."""
    df = ngdf.loc[year_start:year_end, [state]]
    df = df.rename(columns={state:'Production'})
    return hv.Curve(df, kdims=["Date"], vdims=["Production"])#.opts()


NGProduction("Other", 2000).opts(width=800)



# ### DynamicMap
#
# Using DynamicMap allows us to explore parameter space similar to widgets. We can manipulate the inputs of the function we just created using sliders and select boxes.
#
# You can export these if you make them a HoloMaps instead, which precomputes everything.
#
#
# - make a function
# - declare parameters space
# - display

dmap = hv.DynamicMap(NGProduction, kdims=["state", "year_start", "year_end"])
dmap = dmap.redim.range(year_start=(1991, 2015), year_end=(1995, 2020))
dmap = dmap.redim.values(state=ngdf.columns)
dmap.redim.default(state="Other", year_start=1991, year_end=2020).opts(width=500)

# __Note:__ We used `.range` for sliders and `.values` for dropdown menus or define sliders with discrecte values.

# <div class="alert alert-success">
# <h2>Exercise</h2>
#
# Make a dynamic map for the smart meter data
#     
# ```python
# df_smartmeter = pd.read_csv("../../data/processed/smartmeter/block_0.csv", parse_dates=['day'], index_col=['day'])
# print(df.head())
#
# def smart_meter_curve(LCLid, start_date=2012, end_date=2015):
#     # YOUR CODE HERE
#     return curve_hh
#
# dmap = hv.DynamicMap(
#     smart_meter_curve, 
#     kdims=["LCLid", "start_date", "end_date"])
# dmap = dmap.redim.range(
#     start_date=(?, ?), end_date=(?, ?) # YOUR CODE HERe
# )
# dmap = dmap.redim.values(LCLid=df_smartmeter.LCLid.unique())
# dmap.redim.default(state="Other", year_start=2013, year_end=2014)
# ```
#
#
# <details>
# <summary><b>→ Hints</b></summary>
#
# * Inside the function you will need to
#     * Select a household e.g. `df=df[df.LCLid==LCLid]`
#     * Select a time e.g. `.loc['2012':'2013']`
#     * Then you will need to make a curve, see the last exercise
# * Fill those question marks with the start and end dates
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
# df_smartmeter = pd.read_csv("../../data/processed/smartmeter/block_0.csv", parse_dates=['day'], index_col=['day'])
# print(df.head())
#
# def smart_meter_curve(LCLid, start_date=2012, end_date=2015):
#     household = df_smartmeter[df_smartmeter.LCLid==LCLid].loc[str(start_date):str(end_date)]
#     curve_hh = hv.Curve(household, kdims=["day"], vdims=["energy_sum"])
#     curve_hh = curve_hh.redim.unit(energy_sum="kWh")
#     return curve_hh
#
# dmap = hv.DynamicMap(smart_meter_curve, kdims=["LCLid", "start_date", "end_date"])
# dmap = dmap.redim.range(
#     start_date=(2012, 2015), end_date=(2012, 2015)
# )
# dmap = dmap.redim.values(LCLid=df_smartmeter.LCLid.unique())
# dmap.redim.default(state="Other", year_start=2013, year_end=2014)
# ```
#
# </details>
#
# </div>





# ## Gridded Data
# Gridded datasets contain the value of a certain variable in multiple dimensions. For instance, the temperature at various latitudes and longtitudes. Visualising 2D data is simple, but it can get more complicated when dealing with higher dimensions. Holoviews allows us to easily explore this type of data using tools such as sliders.

# We are going to use DeepRock-SR dataset which contains digital rock images. We are going to load a 3D image and explore it.

# <div class='alert alert-warning'>We are going to use `h5py` library to read the file.</div>

import h5py

path = "../../data/processed/deep-rock-sr/DeepRockSR-3D/coal3D/coal3D_valid_LR_default_X4/0801x4.mat"
fp = h5py.File(path, "r")
mat = fp.get("temp")

path = "../../data/processed/deep-rock-sr/DeepRockSR-3D/coal3D/coal3D_valid_HR/0801.mat"
fp = h5py.File(path, "r")
mat = fp.get("temp")



# First we need to convert it to a numpy array.

mat = np.array(mat)
mat.shape

# Just to see the content of the file, let's plot it with matplotlib first.

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 16))
for i in range(100):
    plt.subplot(5, 5, i//5 + 1)
    plt.imshow(mat[:, :, i], "gray")

# Now we need to convert the data into a form which is easy to understand for holoviews. First, we create x, y, and z coordinates.

# Then convert the 3D data into a holoviews dataset

nx, ny, nz = mat.shape
rock = hv.Dataset((np.arange(nx), np.arange(ny), np.arange(nz), mat), ["x", "y", "z"], ["value"])
rock = rock.clone(datatype=['xarray'])
rock.data

from holoviews import opts
opts.defaults(
    opts.GridSpace(shared_xaxis=True, shared_yaxis=True),
    opts.Image(cmap='Gray', width=400, height=400),
    opts.Labels(text_color='white', text_font_size='8pt', text_align='left', text_baseline='bottom'),
    opts.Path(color='white'),
    opts.Spread(width=600),
    opts.Overlay(show_legend=False))

# Data has three dimensions. We can easily visualise two using an image, and use a slider to change the third dimension.

rock.to(hv.Image, ['x', 'y'], datatype=["xarray"], dynamic=True).hist()

# To make it more interesting we can create three plots and each having one of the dimensions on the slider.
#
# Here's we use the [Turbo](https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html) colormap which is often better than viridis or plasma for highlighting differences

import bokeh.palettes as pal

# %%opts Image {+axiswise} [xaxis=None yaxis=None width=200 height=200]
cmap = pal.Turbo256
hmx = rock.to(hv.Image, ['z', 'y'], dynamic=True).opts(cmap=cmap)
hmy = rock.to(hv.Image, ['z', 'x'], dynamic=True).opts(cmap=cmap)
hmz = rock.to(hv.Image, ['x', 'y'], dynamic=True).opts(cmap=cmap)
(hmx + hmy + hmz).redim.range(value=(mat.min(), mat.max()))

# We can also aggregate on of the dimensions and turn the data into 2D. In the example below `x` dimension is removed by the average `x` and replaced by an average value in that direction.

# +
# hv.Image(rock.reduce(x=np.mean)).opts(cmap=cmap)
# -

# ## [Advanced] Dashboards
# We learned about using `DynamicMap` to create tools which helps us explore the data. But the tools we had access to were limited to a dropdown menu and a slider. What if we need more advanced tools. There is a package called __Panel__ developed by the same team as holoviews which specialised in creating dashboards. Here we will see a few examples of panel to see how we can use holoviews and panel to create dashboards.

import panel as pn

# We are going to use a data set containing production rate of a few wells in Buchan Oil Field.

# +
import geopandas as gpd
import matplotlib.pyplot as plt

df = gpd.read_file(
    "../../data/processed/daily_buchan_prod_data/Daily_Buchan_Production_Data.gpkg"
)
# -

df.head()

df.tail()

df["DATE_"] = pd.to_datetime(df["DATE_"])
df.sort_values(by=["WELLID", "DATE_"], inplace=True)
df.index = df["DATE_"]

wells = df["WELLID"].unique()
x = df.loc[df["WELLID"] == wells[0], "DATE_"]
y = df.loc[df["WELLID"] == wells[0], "DAILY_WELL_DRY_OIL_BBL"]
plt.plot(x, y)

df.describe()

# ### Using functions
#
# There are multiple ways to build dashboards, we will use functions. There is also another way of achieving the same result. We can use panel's widgets and decorator to attach the widget directly to a function.
#
# Please note that this section use some more advanced programming concepts like decorators and callbacks.

# First, we need to create widget elements using which we will choose the value.

well_id = pn.widgets.Select(options=df["WELLID"].unique().tolist())
product = pn.widgets.RadioButtonGroup(options=["Oil", "Gas", "Water"])
product


# Then we will define a callback function which will generate the plot. To attach this function to the widgets we use a decorator.

# <div class='alert alert-warning' style='font-size=120%'>
#     Decorators are functions which take another function as an input and then modify it and return a new function. To learn more about decorators, what they are and why you would use them check out this <a href='https://www.datacamp.com/community/tutorials/decorators-python'>page</a>.
# </div>

# In the decorator we need to apecify how the inputs of the function are connected to the widgets.

@pn.depends(well_id=well_id.param.value, product=product.param.value)
def production_callback(well_id, product):
    if product.lower() == "oil":
        col = "DAILY_WELL_DRY_OIL_BBL"
    elif product.lower() == "gas":
        col = "PRODUCED_GAS_GAS_MMCF"
    else:
        col = "PRODUCED_WATER_BBL"

    x = df.loc[df["WELLID"] == well_id, "DATE_"]
    y = df.loc[df["WELLID"] == well_id, col]
    prod_plot = hv.Curve((x, y), kdims=["Date"], vdims=["Production"])
    return prod_plot


# Then, we create a dynamic map using the callback function.

dmap = hv.DynamicMap(production_callback)

# We also need to create a widget box which contains all the widgets.

widget_box = pn.WidgetBox("#### Production Explorer", product, well_id)
widget_box.width = 200
widget_box.height = 150

# And now we are ready to see the result

pn.Row(widget_box, dmap)


# This was a simple example, but using the same techniques we can generate more complex plots and widgets as well.

# Let's add a second plot which shows the location of the wells as well.

# +
@pn.depends(well_id=well_id.param.value)
def wells_callback(well_id):
    # shows the location of all wells
    locations = hv.Scatter(df[["X", "Y", "WELLID"]].groupby("WELLID").mean()).opts(
        xlim=(-0.02, 0.04), ylim=(57.89, 57.906), size=10, marker="x", color="k"
    )

    # shows the location of selected well
    target_location = hv.Scatter(
        df.loc[df["WELLID"] == well_id, ["X", "Y", "WELLID"]].groupby("WELLID").mean()
    ).opts(size=10, color="r")

    return locations * target_location


dmap2 = hv.DynamicMap(wells_callback)

# +
# %opts Scatter [height=350 width=350]
widget_box = pn.WidgetBox('#### Production Explorer', product,well_id)


pn.Column(widget_box,pn.Row(dmap2,dmap))
# -

# __The example below plots the production rate of products against each other for each year.__

# +
product1 = pn.widgets.RadioButtonGroup(options=["Oil", "Gas", "Water"], value="Oil")
product2 = pn.widgets.RadioButtonGroup(options=["Oil", "Gas", "Water"], value="Gas")
year = pn.widgets.Player(start=1982, end=2010, value=1990, loop_policy="loop")

# We can also use a slider to select the year
# year = pn.widgets.IntSlider(start=1982, end=2010, value=1990)


def get_column_name(product):
    if product.lower() == "oil":
        col = "DAILY_WELL_DRY_OIL_BBL"
    elif product.lower() == "gas":
        col = "PRODUCED_GAS_GAS_MMCF"
    else:
        col = "PRODUCED_WATER_BBL"
    return col


@pn.depends(
    product1=product1.param.value, product2=product2.param.value, year=year.param.value
)
def production_callback2(product1, product2, year):
    col1 = get_column_name(product1)
    col2 = get_column_name(product2)
    df_year = df[df.index.year==year]
    plots = hv.Scatter([])
    fields = df["FIELDWELL"].unique()
    cmap = palettes.Category20[len(fields)]
    for i, fld in enumerate(fields):
        plots *= hv.Scatter(
            df_year.loc[df_year["FIELDWELL"] == fld], kdims=[col1], vdims=[col2]
        ).opts(
            xlabel=product1.upper(),
            ylabel=product2.upper(),
            title=f"Year: {year}",
            logx=True,
            logy=True,
            color=cmap[i],
            size=5,
            alpha=0.5,
            bgcolor="black",
            tools=["hover"],
        )

    return plots
# -



dmap = hv.DynamicMap(production_callback2)

widget_box = pn.WidgetBox(
    "#### Production Explorer",
    pn.WidgetBox("__X Axis__", product1, align="center"),
    pn.WidgetBox("__Y Axis__", product2, align="center"),
    year,
    align="center",
)
# Uncomment the line below if you want to manually change the size of widget-box
# widget_box.width = 300
# widget_box.height = 300

# %opts Scatter [width=500 height=500]
pn.Column(widget_box,dmap)

# <div class='alert alert-warning'>If the plot looks empty, try zooming out. As the values change sometimes the move outside of the initial range.</div>

# `Panel` has a long list of different widgets you can use in your dashboard. You can find the full list [here](https://panel.holoviz.org/user_guide/Widgets.html).

#

# # Further Reading
#
# This notebook was based on open source tutorials especially https://github.com/ioam/jupytercon2017-holoviews-tutorial
#
# Other are listed below, along with further reading:
#
# - [HoloViews getting started](https://holoviews.org/getting_started/Introduction.html)
# - [Holoviews Examples](http://holoviews.org/Examples/index.html)
# - [Panel Dashboard Examples](https://panel.holoviz.org/gallery/index.html)
# - https://www.kaggle.com/ukveteran/holoviews-masterclass-jma
# - https://github.com/ioam/jupytercon2017-holoviews-tutorial
# - https://www.youtube.com/watch?v=cMXKE0nB8k4
#


