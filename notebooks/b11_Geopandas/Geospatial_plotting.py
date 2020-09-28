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
#     display_name: py37_pytorch
#     language: python
#     name: conda-env-py37_pytorch-py
# ---

# # Geospatial plotting

# +
# %matplotlib inline

import pandas as pd
import geopandas as gpd
# -

# Geospatial data is often available from specific GIS file formats or data stores, like ESRI shapefiles, GeoJSON files, geopackage files, PostGIS (PostgreSQL) database, ...
#
# We can use the GeoPandas library to read many of those GIS file formats (relying on the `fiona` library under the hood, which is an interface to GDAL/OGR), using the `geopandas.read_file` function.
#
# For example, let's start by reading a shapefile with all the countries of the world (adapted from http://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/, and inspect the data:

countries = gpd.read_file(
    "zip://../../data/processed/geo/ne_110m_admin_0_countries.zip"
)

countries.head()

countries.plot(figsize=(16, 8))

# What can we observe:
#
# - Using `.head()` we can see the first rows of the dataset, just like we can do with Pandas.
# - There is a 'geometry' column and the different countries are represented as polygons
# - We can use the `.plot()` method to quickly get a *basic* visualization of the data

# ## What's a GeoDataFrame?
#
# We used the GeoPandas library to read in the geospatial data, and this returned us a `GeoDataFrame`:

# A GeoDataFrame contains a tabular, geospatial dataset:
#
# * It has a **'geometry' column** that holds the geometry information (or features in GeoJSON).
# * The other columns are the **attributes** (or properties in GeoJSON) that describe each of the geometries
#
# Such a `GeoDataFrame` is just like a pandas `DataFrame`, but with some additional functionality for working with geospatial data:
#
# * A `.geometry` attribute that always returns the column with the geometry information (returning a GeoSeries). The column name itself does not necessarily need to be 'geometry', but it will always be accessible as the `.geometry` attribute.
# * It has some extra methods for working with spatial data (area, distance, buffer, intersection, ...), which we will see in later notebooks

countries.geometry

type(countries.geometry)

countries.geometry.area

# **It's still a DataFrame**, so we have all the pandas functionality available to use on the geospatial dataset, and to do data manipulations with the attributes and geometry information together.
#
# For example, we can calculate average population number over all countries (by accessing the 'pop_est' column, and calling the `mean` method on it):

countries["pop_est"].mean()

# Or, we can use boolean filtering to select a subset of the dataframe based on a condition:

africa = countries[countries["continent"] == "Africa"]

africa.plot()

# <div class="alert alert-info" style="font-size:120%">
#
# **REMEMBER:** <br>
#
# * A `GeoDataFrame` allows to perform typical tabular data analysis together with spatial operations
# * A `GeoDataFrame` (or *Feature Collection*) consists of:
#     * **Geometries** or **features**: the spatial objects
#     * **Attributes** or **properties**: columns with information about each spatial object
#
# </div>

# ## Geometries: Points, Linestrings and Polygons
#
# Spatial **vector** data can consist of different types, and the 3 fundamental types are:
#
# ![](img/simple_features_3_text.svg)
#
# * **Point** data: represents a single point in space.
# * **Line** data ("LineString"): represents a sequence of points that form a line.
# * **Polygon** data: represents a filled area.
#
# And each of them can also be combined in multi-part geometries (See https://shapely.readthedocs.io/en/stable/manual.html#geometric-objects for extensive overview).

# For the example we have seen up to now, the individual geometry objects are Polygons:

print(countries.geometry[2])

# Let's import some other datasets with different types of geometry objects.

cities = gpd.read_file(
    "zip://../../data/processed/geo/ne_110m_populated_places.zip"
)

print(cities.geometry[0])

# And a dataset of rivers in the world where each river is a (multi-)line:

rivers = gpd.read_file(
    "zip://../../data/processed/geo/ne_50m_rivers_lake_centerlines.zip"
)

print(rivers.geometry[0])

rivers.geometry[0]

# ### The `shapely` library
#
# The individual geometry objects are provided by the [`shapely`](https://shapely.readthedocs.io/en/stable/) library

type(countries.geometry[0])

# To construct one ourselves:

from shapely.geometry import Point, Polygon, LineString

p = Point(0, 0)

print(p)

polygon = Polygon([(1, 1), (2, 2), (2, 1)])

polygon.area

polygon.distance(p)

# <div class="alert alert-info" style="font-size:120%">
#
# **REMEMBER**: <br>
#
# Single geometries are represented by `shapely` objects:
#
# * If you access a single geometry of a GeoDataFrame, you get a shapely geometry object
# * Those objects have similar functionality as geopandas objects (GeoDataFrame/GeoSeries). For example:
#     * `singleShapelyObject.distance(other_point)` -> distance between two points
#     * `geodataframe.distance(other_point)` ->  distance for each point in the geodataframe to the other point
#
# </div>

# ## Plotting our different layers together

ax = countries.plot(edgecolor="k", facecolor="none", figsize=(15, 10))
rivers.plot(ax=ax)
cities.plot(ax=ax, color="red")
ax.set(xlim=(-20, 60), ylim=(-40, 40))

# # Coordinate reference systems

countries = gpd.read_file(
    "zip://../../data/processed/geo/ne_110m_admin_0_countries.zip"
)
cities = gpd.read_file(
    "zip://../../data/processed/geo/ne_110m_populated_places.zip"
)
rivers = gpd.read_file(
    "zip://../../data/processed/geo/ne_50m_rivers_lake_centerlines.zip"
)

# ## Coordinate reference systems
#
# Up to now, we have used the geometry data with certain coordinates without further wondering what those coordinates mean or how they are expressed.
#
# > The **Coordinate Reference System (CRS)** relates the coordinates to a specific location on earth.
#
# For a nice in-depth explanation, see https://docs.qgis.org/2.8/en/docs/gentle_gis_introduction/coordinate_reference_systems.html

# ### Geographic coordinates
#
# > Degrees of latitude and longitude.
# >
# > E.g. 48°51′N, 2°17′E
#
# The most known type of coordinates are geographic coordinates: we define a position on the globe in degrees of latitude and longitude, relative to the equator and the prime meridian. 
# With this system, we can easily specify any location on earth. It is used widely, for example in GPS. If you inspect the coordinates of a location in Google Maps, you will also see latitude and longitude.
#
# **Attention!**
#
# in Python we use (lon, lat) and not (lat, lon)
#
# - Longitude: [-180, 180]{{1}}
# - Latitude: [-90, 90]{{1}}

# ### Projected coordinates
#
# > `(x, y)` coordinates are usually in meters or feet
#
# Although the earth is a globe, in practice we usually represent it on a flat surface: think about a physical map, or the figures we have made with Python on our computer screen.
# Going from the globe to a flat map is what we call a *projection*.
#
# ![](img/projection.png)
#
# We project the surface of the earth onto a 2D plane so we can express locations in cartesian x and y coordinates, on a flat surface. In this plane, we then typically work with a length unit such as meters instead of degrees, which makes the analysis more convenient and effective.
#
# However, there is an important remark: the 3 dimensional earth can never be represented perfectly on a 2 dimensional map, so projections inevitably introduce distortions. To minimise such errors, there are different approaches to project, each with specific advantages and disadvantages.
#
# Some projection systems will try to preserve the area size of geometries, such as the Albers Equal Area projection. Other projection systems try to preserve angles, such as the Mercator projection, but will see big distortions in the area. Every projection system will always have some distortion of area, angle or distance.
#
# <table><tr>
# <td> <img src="img/projections-AlbersEqualArea.png"/> </td>
# <td> <img src="img/projections-Mercator.png"/> </td>
# </tr>
# <tr>
# <td> <img src="img/projections-Robinson.png"/> </td>
# </tr></table>

# **Projected size vs actual size (Mercator projection)**:
#
# ![](img/mercator_projection_area.gif)

# ## Coordinate Reference Systems in Python / GeoPandas

# A GeoDataFrame or GeoSeries has a `.crs` attribute which holds (optionally) a description of the coordinate reference system of the geometries:

countries.crs

# For the `countries` dataframe, it indicates that it uses the EPSG 4326 / WGS84 lon/lat reference system, which is one of the most used for geographic coordinates.
#
#
# It uses coordinates as latitude and longitude in degrees, as can you be seen from the x/y labels on the plot:

countries.plot()

# The `.crs` attribute is given as a dictionary. In this case, it only indicates the EPSG code, but it can also contain the full "proj4" string (in dictionary form).
#
# Possible CRS representation:
#
# - **`proj4` string**  
#   
#   Example: `+proj=longlat +datum=WGS84 +no_defs`
#
#   Or its dict representation: `{'proj': 'longlat', 'datum': 'WGS84', 'no_defs': True}`
#
# - **EPSG code**
#   
#   Example: `EPSG:4326` = WGS84 geographic CRS (longitude, latitude)
#   
# - Well-Know-Text (WKT) representation (better support coming with PROJ6 in the next GeoPandas version)
#
# See eg https://epsg.io/4326
#
# Under the hood, GeoPandas uses the `pyproj` / `PROJ` libraries to deal with the re-projections.
#
# For more information, see also http://geopandas.readthedocs.io/en/latest/projections.html.

# ### Transforming to another CRS
#
# We can convert a GeoDataFrame to another reference system using the `to_crs` function. 
#
# For example, let's convert the countries to the World Mercator projection (http://epsg.io/3395):

# remove Antartica, as the Mercator projection cannot deal with the poles
countries = countries[(countries["name"] != "Antarctica")]
countries.plot()
countries.crs

countries_mercator = countries.to_crs(epsg=3395)  # or .to_crs({'init': 'epsg:3395'})

countries_mercator.plot()

# Note the different scale of x and y.

# ### Why use a different CRS?
#
# There are sometimes good reasons you want to change the coordinate references system of your dataset, for example:
#
# - Different sources with different CRS -> need to convert to the same crs
#
#     ```python
#     df1 = gpd.read_file(...)
#     df2 = gpd.read_file(...)
#
#     df2 = df2.to_crs(df1.crs)
#     ```
#
# - Mapping (distortion of shape and distances)
#
# - Distance / area based calculations -> ensure you use an appropriate projected coordinate system expressed in a meaningful unit such as metres or feet (not degrees).
#
# <div class="alert alert-info" style="font-size:120%">
#
# **ATTENTION:**
#
# All the calculations that happen in geopandas and shapely assume that your data is in a 2D cartesian plane, and thus the result of those calculations will only be correct if your data is properly projected.
#
# </div>
#
#

# # Spatial relationships and operations

countries = gpd.read_file(
    "zip://../../data/processed/geo/ne_110m_admin_0_countries.zip"
)
cities = gpd.read_file(
    "zip://../../data/processed/geo/ne_110m_populated_places.zip"
)
rivers = gpd.read_file(
    "zip://../../data/processed/geo/ne_50m_rivers_lake_centerlines.zip"
)

# ## Spatial relationships
#
# An important aspect of geospatial data is that we can look at *spatial relationships*: how two spatial objects relate to each other (whether they overlap, intersect, contain, .. one another).
#
# The topological, set-theoretic relationships in GIS are typically based on the DE-9IM model. See https://en.wikipedia.org/wiki/Spatial_relation for more information.
#
# ![](img/TopologicSpatialRelarions2.png)
# (Image by [Krauss, CC BY-SA 3.0](https://en.wikipedia.org/wiki/Spatial_relation#/media/File:TopologicSpatialRelarions2.png))

# ### Relationships between individual objects

# Let's first create some small toy spatial objects:
#
# A polygon <small>(note: we use `.squeeze()` here to to extract the shapely geometry object from the GeoSeries of length 1)</small>:

belgium = countries.loc[countries["name"] == "Belgium", "geometry"].squeeze()
belgium

# Two points:

paris = cities.loc[cities["name"] == "Paris", "geometry"].squeeze()
brussels = cities.loc[cities["name"] == "Brussels", "geometry"].squeeze()
brussels

# And a linestring, a line from Paris to Brussels:

# +
from shapely.geometry import LineString

line = LineString([paris, brussels])
line
# -

# Let's visualize those 4 geometry objects together (I only put them in a GeoSeries to easily display them together with the geopandas `.plot()` method):

gpd.GeoSeries([belgium, paris, brussels, line]).plot(cmap="tab10")

# You can recognize the abstract shape of Belgium.
#
# Brussels, the capital of Belgium, is thus located within Belgium. This is a spatial relationship, and we can test this using the individual shapely geometry objects as follow:

brussels.within(belgium)

# And using the reverse, Belgium contains Brussels:

belgium.contains(brussels)

# On the other hand, Paris is not located in Belgium:

belgium.contains(paris)

paris.within(belgium)

# The straight line we draw from Paris to Brussels is not fully located within Belgium, but it does intersect with it:

belgium.contains(line)

line.intersects(belgium)

# ### Spatial relationships with GeoDataFrames
#
# The same methods that are available on individual `shapely` geometries as we have seen above, are also available as methods on `GeoSeries` / `GeoDataFrame` objects.
#
# For example, if we call the `contains` method on the world dataset with the `paris` point, it will do this spatial check for each country in the `world` dataframe:

countries.contains(paris)

# Because the above gives us a boolean result, we can use that to filter the dataframe:

countries[countries.contains(paris)]

# Or (clue)
mask = cities.within(brussels)
cities[mask]

# And indeed, France is the only country in the world in which Paris is located.

# Another example, extracting the linestring of the Amazon river in South America, we can query through which countries the river flows:

amazon = rivers[rivers["name"] == "Amazonas"].geometry.squeeze()
amazon

countries[countries.crosses(amazon)]  # or .intersects

# <div class="alert alert-info" style="font-size:120%">
#
# **REFERENCE**:
#
# Overview of the different functions to check spatial relationships (*spatial predicate functions*):
#
# * `equals`
# * `contains`
# * `crosses`
# * `disjoint`
# * `intersects`
# * `overlaps`
# * `touches`
# * `within`
# * `covers`
#
#
# See https://shapely.readthedocs.io/en/stable/manual.html#predicates-and-relationships for an overview of those methods.
#
# See https://en.wikipedia.org/wiki/DE-9IM for all details on the semantics of those operations.
#
# </div>

# ## Exercise
#
# We will work with datasets about the city of Paris, starting with the following datasets:
#
# - The administrative districts of Paris: `paris_districts_utm.geojson`
# - Information about the public bicycle sharing system in Paris (vélib): `data/paris_bike_stations_mercator.gpkg`
#

districts = gpd.read_file(
    "../../data/processed/geo/paris_districts.geojson"
).to_crs(epsg=2154)
stations = gpd.read_file(
    "../../data/processed/geo/paris_bike_stations.geojson"
).to_crs(epsg=2154)

# <div class="alert alert-success">
#
# **The Eiffel Tower**
#
# The Eiffel Tower is an iron lattice tower built in the 19th century, and is probably the most iconic view of Paris.
#
# The location of the Eiffel Tower is: x of 648237.3 and y of 6862271.9.
#
# 1. Create a Shapely point object with the coordinates of the Eiffel Tower and assign it to a variable called `eiffel_tower`. Print the result.
# 2. Check if the Eiffel Tower is located within the Montparnasse district (provided).
# 3. Check if there are any bike stations within the Montparnasse district.
# 4. Calculate the distance all bike stations and the Eiffel Tower (note: in this case, the distance is returned in meters).
#
#
# <details><summary>Hints</summary>
#
# * The `Point` class is available in the `shapely.geometry` submodule
# * Creating a point can be done by passing the x and y coordinates to the `Point()` constructor.
# * The `within()` method checks if the object is located within the passed geometry (used as `geometry1.within(geometry2)`).
# * The `contains()` method checks if the object contains the passed geometry (used as `geometry1.contains(geometry2)`).
# * To calculate the distance between two geometries, the `distance()` method of one of the geometries can be used.
# * Always have the dataframe on the left of the statement, if possible
#
# </details>
#
# </div>

# +
# Part 1

# +
# Part 2

# +
# Part 3

# +
# Part 4
# -

# <details><summary>Solution</summary>
# <b>Part 1</b>
#     
# ```Python
# eiffel_tower = Point(648237.3, 6862271.9)
# print(eiffel_tower)
# ```
#
# <b>Part 2</b>
#     
# ```Python
# montparnasse = districts[districts['district_name']=='Montparnasse'].geometry.squeeze()
# montparnasse.contains(eiffel_tower)
# ```
#
# <b>Part 3</b>    
#
# ```Python
# stations[stations.within(montparnasse)]
# ```
#
# <b>Part 4</b>    
#     
# ```Python
# stations[stations.within(montparnasse)].distance(eiffel_tower)
# ```
#
# </details>

# ## Spatial operations
#
# Next to the spatial predicates that return boolean values, Shapely and GeoPandas also provide operations that return new geometric objects.
#
# **Binary operations:**
#
# <table><tr>
# <td> <img src="img/spatial-operations-base.png"/> </td>
# <td> <img src="img/spatial-operations-intersection.png"/> </td>
# </tr>
# <tr>
# <td> <img src="img/spatial-operations-union.png"/> </td>
# <td> <img src="img/spatial-operations-difference.png"/> </td>
# </tr></table>
#
# **Buffer:**
#
# <table><tr>
# <td> <img src="img/spatial-operations-buffer-point1.png"/> </td>
# <td> <img src="img/spatial-operations-buffer-point2.png"/> </td>
# </tr>
# <tr>
# <td> <img src="img/spatial-operations-buffer-line.png"/> </td>
# <td> <img src="img/spatial-operations-buffer-polygon.png"/> </td>
# </tr></table>
#
#
# See https://shapely.readthedocs.io/en/stable/manual.html#spatial-analysis-methods for more details.

# For example, using the toy data from above, let's construct a buffer around Brussels (which returns a Polygon):
#
# Note: 1 degree is around 111km

gpd.GeoSeries([belgium, brussels, brussels.buffer(1)]).plot(
    alpha=0.5, cmap="tab10"
)

# and now take the intersection, union or difference of those two polygons:

brussels.buffer(1).intersection(belgium)

brussels.buffer(1).union(belgium)

brussels.buffer(1).difference(belgium)

# Another useful method is the `unary_union` attribute, which converts the set of geometry objects in a GeoDataFrame into a single geometry object by taking the union of all those geometries.
#
# For example, we can construct a single object for the Africa continent:

africa_countries = countries[countries["continent"] == "Africa"]
africa_countries.plot()

africa = africa_countries.unary_union
africa



print(str(africa)[:1000])

# <div class="alert alert-info" style="font-size:120%">
#
# **REMEMBER**:
#
# GeoPandas (and Shapely for the individual objects) provides a whole lot of basic methods to analyse the geospatial data (distance, length, centroid, boundary, convex_hull, simplify, transform, ....), much more than the few that we can touch in this tutorial.
#
#
# * An overview of all methods provided by GeoPandas can be found here: http://geopandas.readthedocs.io/en/latest/reference.html
#
#
# </div>
#
#

# # Spatial joins

# + [markdown] slideshow={"slide_type": "fragment"}
# Goals of this notebook:
#
# - Based on the `countries` and `cities` dataframes, determine for each city the country in which it is located.
# - To solve this problem, we will use the the concept of a 'spatial join' operation: combining information of geospatial datasets based on their spatial relationship.
# -

# To illustrate the concept of joining the information of two dataframes with pandas, let's take a small subset of our `cities` and `countries` datasets: 

cities2 = cities[cities["name"].isin(["Bern", "Brussels", "London", "Paris"])].copy()
cities2["iso_a3"] = ["CHE", "BEL", "GBR", "FRA"]

cities2

countries2 = countries[["iso_a3", "name", "continent"]]
countries2.head()

# We added a 'iso_a3' column to the `cities` dataset, indicating a code of the country of the city. This country code is also present in the `countries` dataset, which allows us to merge those two dataframes based on the common column.
#
# Joining the `cities` dataframe with `countries` will transfer extra information about the countries (the full name, the continent) to the `cities` dataframe, based on a common key:

cities2.merge(countries2, on="iso_a3")

# **But**, for this illustrative example, we added the common column manually, it is not present in the original dataset. However, we can still know how to join those two datasets based on their spatial coordinates.

# In this case, we know that each of the cities is located *within* one of the countries, or the other way around that each country can *contain* multiple cities. We can test such relationships using the methods we have seen previously.

france = countries.loc[countries["name"] == "France", "geometry"].squeeze()

cities.within(france)

# The above gives us a boolean series, indicating for each point in our `cities` dataframe whether it is located within the area of France or not.  
# Because this is a boolean series as result, we can use it to filter the original dataframe to only show those cities that are actually within France:

cities[cities.within(france)]

# We could now repeat the above analysis for each of the countries, and add a column to the `cities` dataframe indicating this country. However, that would be tedious to do manually, and is also exactly what the spatial join operation provides us.
#
# <font color='red'>*(note: the above result is incorrect, but this is just because of the coarse-ness of the countries dataset)*</font>

# + [markdown] slideshow={"slide_type": "slide"}
# ## Spatial join operation
#
# <div class="alert alert-info" style="font-size:120%">
#     
# **SPATIAL JOIN** = *transferring attributes from one layer to another based on their spatial relationship* <br>
#
#
# Different parts of this operations:
#
# * The GeoDataFrame to which we want add information
# * The GeoDataFrame that contains the information we want to add
# * The spatial relationship we want to use to match both datasets ('intersects', 'contains', 'within')
# * The type of join: left or inner join
#
#
# ![](img/illustration-spatial-join.svg)
#
# </div>

# + [markdown] slideshow={"slide_type": "-"}
# In this case, we want to join the `cities` dataframe with the information of the `countries` dataframe, based on the spatial relationship between both datasets.
#
# We use the [`geopandas.sjoin`](http://geopandas.readthedocs.io/en/latest/reference/geopandas.sjoin.html) function:
# -

joined = gpd.sjoin(cities, countries, op="within", how="left")

joined

joined["continent"].value_counts()

# ## The overlay operation
#
# In the spatial join operation above, we are not changing the geometries itself. We are not joining geometries, but joining attributes based on a spatial relationship between the geometries. This also means that the geometries need to at least overlap partially.
#
# If you want to create new geometries based on joining (combining) geometries of different dataframes into one new dataframe (eg by taking the intersection of the geometries), you want an **overlay** operation.

africa = countries[countries["continent"] == "Africa"]

africa.plot()

cities["geometry"] = cities.buffer(2)

gpd.overlay(africa, cities, how="difference").plot()

# <div class="alert alert-info" style="font-size:120%">
# <b>REMEMBER</b> <br>
#
# * **Spatial join**: transfer attributes from one dataframe to another based on the spatial relationship
# * **Spatial overlay**: construct new geometries based on spatial operation between both dataframes (and combining attributes of both dataframes)
#
# </div>

# # Visualizing spatial data with Python

# ## GeoPandas visualization functionality

# #### Basic plot

countries.plot()

# #### Adjusting the figure size

countries.plot(figsize=(15, 15))

# #### Removing the box / x and y coordinate labels

ax = countries.plot(figsize=(15, 15))
ax.set_axis_off()

# #### Coloring based on column values
#
# Let's first create a new column with the GDP per capita:

countries = countries[(countries["pop_est"] > 0) & (countries["name"] != "Antarctica")]

countries["gdp_per_cap"] = countries["gdp_md_est"] / countries["pop_est"] * 100

# and now we can use this column to color the polygons:

ax = countries.plot(figsize=(15, 15), column="gdp_per_cap")
ax.set_axis_off()

ax = countries.plot(
    figsize=(15, 15), column="gdp_per_cap", scheme="quantiles", legend=True
)
ax.set_axis_off()

# #### Combining different dataframes on a single plot
#
# The `.plot` method returns a matplotlib Axes object, which can then be re-used to add additional layers to that plot with the `ax=` keyword:

ax = countries.plot(figsize=(15, 15))
cities.plot(ax=ax, color="red", markersize=10)
ax.set_axis_off()

ax = countries.plot(edgecolor="k", facecolor="none", figsize=(15, 10))
rivers.plot(ax=ax)
cities.plot(ax=ax, color="C1")
ax.set(xlim=(-20, 60), ylim=(-40, 40))

# <div class='alert alert-info'>This notebook is based on a series of notebook from <strong> <a href='https://github.com/jorisvandenbossche/geopandas-tutorial'>Introduction to geospatial data analysis with GeoPandas and the PyData stack</a></strong>.</div>

# # Further Reading
# - [Introduction to geospatial data analysis with GeoPandas and the PyData stack](https://github.com/jorisvandenbossche/geopandas-tutorial)
# - [Geopandas Documentation](https://geopandas.org/index.html)
# - [Geopandas Examples](https://geopandas.org/gallery/index.html)
# - [Geoplot Documentation](https://residentmario.github.io/geoplot/index.html)
# - [Visualizing Geospatial Data in Python](https://towardsdatascience.com/visualizing-geospatial-data-in-python-e070374fe621)
