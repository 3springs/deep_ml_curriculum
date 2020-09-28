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

# # Pandas

# +
import pandas as pd
import pandas_profiling as pp
import numpy as np
import warnings

warnings.simplefilter("ignore")
# -

# ## Read Data

# Pandas has the ability to read data from various formats, including:
# - CSV
# - Excel
# - Html
# - Json
# - Feather
# - Parquet

# Let's start by reading a table from a csv file. Pandas puts the data in an object known as a `DataFrame`.<br>
# The data we are using here is air emissions from industrial facilities in Queensland for the 2005/2006 inventory year taken from [data.gov.au](http://data.gov.au).

# !head -n 3 "../../data/processed/Emission/npi-2006-qld-air-total-emissions.csv"

df = pd.read_csv("../../data/processed/Emission/npi-2006-qld-air-total-emissions2.csv", index_col='index')

# The `DataFrame` is now stored in variable `df`. We can print it:

df

# Similarly, using the respective read method you can read the tables from other file formats supported by Pandas. <br>
# e.g. `pd.read_excel()` 

# Pandas can also read the data from SQL database and put them directly into a pandas data frame. To do that, we need to pass in the query and the connection object.

# +
import sqlite3

# create a connection to database
conn = sqlite3.connect("../01 SQL/Sales.db")

# write a query
query = """
SELECT * from Customers
LIMIT 5
"""

pd.read_sql(query, conn)
# -

# __Tip:__ Sometimes you just want to quickly copy a portion of a dataset from a webpage or excel into a notebook. An easy way to do that is to copy the data from the source and then use `pd.read_clipboard()`. This method will create a pandas data frame from the data you copied. Note that this method only works if the notebook is running on your local machine (not on an external server).

# ## Basic Analysis
# - info
# - describe
# - pandas profiling
# - value_counts()
#

df = pd.read_csv("../../data/processed/Emission/npi-2006-qld-air-total-emissions2.csv", index_col='index')
df

# We can use `.head()` and `.tail()` to view only top or bottom rows of the table.

# top rows
df.head()

# bottom rows
df.tail()

# __Note:__ You can specify how many rows from the top or bottom of the table you want by passing in a number.<br>
# e.g. `df.head(10)` or `df.tail(3)`

# We can use `.columns` to get a list of column names.

df.columns

# Using `.info()` method you can get a list of the columns and the type of data stored in each.

df.info()

# You can also get some basic statistical analysis of the data using `.describe()` method.

df.describe()

# To get a more detailed analysis of the data in the table we can use a package called `pandas-profiling`. This package extends Pandas and adds detailed reports of the data.
#
# <a href="./profile_qld-air-emissions.html">If you get an error use this pre-made report</a>
#

# +
# profile = pp.ProfileReport(df, title='Pandas Profiling Report')
# profile.to_file("profile_qld-air-emissions.html")
# profile
# -

# ## Subsetting and indexing

# There are multiple ways to get a subset of the data. `loc` is used when we want to specify the names of columns and `iloc` when we want to use the index of the columns.<br>
#

# We can use `loc` by specifying the rows and columns we want by name. e.g. `df.loc[{row(s)}, {column(s) name}]`

# we can get a subset of a single column:

# +
df.loc[:'Q001BOR001-S52', "jurisdiction"]

# Notice we used :NAME for rows which means give me the rows up to NAME
# -

# __Note:__ `loc` has a unique property. Since it is designed to work with names of columns and rows, when you want to get a subset of rows, the result it returns is inclusive. In other words when we passed in `:10` in almost every other python object that means `0` to `9`, but in `loc` it means `0` to `10`. Likewise, `10:20` in `loc` means rows `10` to `20`.

# We can also get a subset of multiple columns by passing a list of columns we want.

df.loc['Q001BOR001-S14':'Q001BOR001-S52', ["Year", "facility_name", "substance", "quantity_in_kg"]]

# `iloc` works similar to `loc`, but instead of names we pass in index of the rows or the columns we want.

# a single column
df.iloc[:10, 5]

# __Note:__ Notice the number of rows here, and compare it with when we used `loc`.

# You can see the number corresponding to a column name here
list(enumerate(df.columns))

# multiple columns
df.iloc[10:20, [1, 9, -3, -1]]

# Another useful method to get a subset of data is using boolean indexing. Booleans are either True or False. If we pass a list of booleans, pandas will return only the rows with True in the list.

mask = df["substance"] == "Oxides of Nitrogen"
mask

# The list above has the value true only on the rows where the substance is "Oxides of Nitrogen". <br>
# __Note:__ you can only see a small portion of the data so the True values might not be visible.

# Now if we pass this as an index into a data frame we only get the rows where substance is "Oxides of Nitrogen".

df[mask]

# This method can also be used with `loc` and `iloc`.

df.loc[
    df["substance"] == "Oxides of Nitrogen",
    ["facility_name", "substance", "quantity_in_kg"],
]

# ## Sorting
# To sort the data in the table based on a certain column we can use the `.sort_values` method. When sorting we need to specify which column we want to sort and whether we want to sort in ascending order or descending order.

df2 = df.sort_values(by="jurisdiction_facility_id", ascending=False)
df2

# __Note:__ many methods return the result as a data frame as well. This allows us to chain these operations to make the code shorter and easier to read.<br>
#  

# Let's sort the table based on the amount of Oxides of Nitrogen only.

df.loc[df["substance"] == "Oxides of Nitrogen"].sort_values(
    by="quantity_in_kg", ascending=False
)

# We can also sort based on multiple columns. To do so we need to pass in the name of the column in a list (in the order we want them to be used for sorting) and also a list to specify whether each column should be ascending or descending.

df.loc[df["substance"] == "Oxides of Nitrogen"].sort_values(
    by=["site_address_postcode", "facility_name"], ascending=[True, False]
)

# ## Data operations
# -merge
# -groupby
# -pivot_table
# -crosstab

# __Groupby:__ It aggregates the data into groups (similar to groupby in SQL). For instance, what if we wanted an average emission of each substance across all the sites? To calculate that we use `.groupby()` method.

# Since we want the average amount of substances, the columns we need will be __substance__ and __quantity_in_kg__.

groups = df[["substance", "quantity_in_kg"]].groupby(by="substance")
groups

name, group = list(groups)[0]
print(name)
group

# But it doesn't show us any tables. The reason is pandas has grouped the data into `DataFrameGroupBy` object and now we need to specify how the values should be aggregated. In this case since we want the average we use `.mean()`.

df[["substance", "quantity_in_kg"]].groupby(by="substance").mean()

# There are other useful aggregation functions such as `.std()` for standard deviation, `.median()` for median, `.count()` for the number of rows in each group, `.sum()` for sum, etc. You can also define your own aggregation function.

agg_func = lambda x: np.sqrt((x ** 2).mean())  # root mean of squares
df[["substance", "quantity_in_kg"]].groupby(by="substance").apply(agg_func)

# <font color='green'>Do you know how to use *__lambda__* functions? If not check out <a href = 'https://www.w3schools.com/python/python_lambda.asp'>this page</a> to learn about them.</font>

# ### Pivot Table
# Another way to represent the data is using pivot tables. You might be familiar with pivot tables in Excel. You can perform the same operations here as well.

# Let's create a pivot table that shows the amount of each substance in every postcode in the dataset.

df.pivot_table(
    index="site_address_postcode",
    columns="substance",
    values="quantity_in_kg",
    aggfunc="mean",
)

# __Note:__ `NaN` stands for Not a Number. In this case it means there was no value available for that cell. This means where you see `NaN` in the table there was no emission recorded for that substance in that specific postcode. This probably means that we can assume the emission was zero. We could let pandas know by passing in `fill_value = 0`. Then, where no value is available pandas put zero instead.
#

df.pivot_table(
    index="site_address_postcode",
    columns="substance",
    values="quantity_in_kg",
    aggfunc="mean",
    fill_value=0,
)



# #### Exercise
#
# <div class="alert alert-success" style="font-size:100%">
# Now to practice what we have learned so far, let's create a table of the total emissions (quantity in kg) of the top 10 substances (most commonly recorded substances in the dataset) for each postcode.<br>
#     
# 1. Find how many times each substance occurs `substance_count`
# 2. Sort `substance_count` and find 10 most common substance `top10`
# 3. Create a seperate dataframe that shows the total weight of each substance per postcode `weight_by_postcode`
# 4. Combine `weight_by_postcode` and `top10` to get the weight by postcode of the top 10 substances 
#     
# <details><summary>Hints</summary>
#     
# 1. use groupby then count
# 2. use .sort_values() then get the first 10 rows
# 3. A pivot table will do this easily
# 4. `weight_by_postcode[top10.index]`
#
# </details>
# <div>



# +
# 1. Find how many times each substance has been recorded (this has been done for you)
# substance_count = df[["site_address_postcode", "substance"]].??

# 2. Sort it and find the substances that have been recorded the most (this has been done for you)
# top10 = substance_count.??

# 3. Create a seperate table that shows the total weight of each substance per postcode (hint: pivot table or groupby)
# pivot = ??

# 4. Combine the tables, to get a subset which only includes the top 10 substances
# pivot_top10 = ??
# -



# #### Solution:

# <details>    
# <summary>
#     <font size="4" color="darkblue"><b>See the solution for Exercise</b></font>
# </summary>
#     
# ```python
# # you can replace site_address_postcode by any other column. Since we are only counting it doesn't matter which column use.
# substance_count = df[["site_address_postcode", "substance"]].groupby(by="substance").count()
#
# # Now sort it, and take the first 10 results
# substance_count.columns = ["Count"] # rename column
# top10 = substance_count.sort_values(by="Count", ascending=False)[:10]
#
# # Create the pivot table
# pivot = df.pivot_table(
#     index="site_address_postcode",
#     columns="substance",
#     values="quantity_in_kg",
#     aggfunc="sum",
# )
#
# # get only the columns for top 10 substances
# pivot_top10 = pivot[top10.index]
# pivot_top10
# ```
# </details>



# Now it's a good time to discuss dealing with missing values in a table.

# ## Missing Values
# There might be missing data in a table. Having `NaN` in the table can cause trouble in the analysis so we need to decide how we are going to deal with it. A few common scenarios are:
# 1. filling the missing values with a number e.g. zero
# 2. removing rows with missing values
# 3. removeing rows with multiple missing values and filling the remaining with a new value

# +
# you can replace site_address_postcode by any other column. Since we are only counting it doesn't matter which column use.
substance_count = df[["site_address_postcode", "substance"]].groupby(by="substance").count()

# Now sort it, and take the first 10 results
substance_count.columns = ["Count"] # rename column
top10 = substance_count.sort_values(by="Count", ascending=False)[:10]

# Create the pivot table
pivot = df.pivot_table(
    index="site_address_postcode",
    columns="substance",
    values="quantity_in_kg",
    aggfunc="sum",
)

# get only the columns for top 10 substances
pivot_top10 = pivot[top10.index]
pivot_top10
# -

# To replace the missing value with a fixed number we can use `.fillna()` method.

dfnew = pivot_top10.fillna(value=0)
dfnew

# __Note:__ When using `fillna` the changes are not saved in the data frame. The default settings only returns the result and keeps the original data frame intact. If you want to save the changes in the same data frame you can pass in `inplace = True`.

# There are other ways to fill the missing values. In some cases you might want to use different values for each column. A common example is using mean or median of a column for the missing values.

fill_values = pivot_top10.mean()
dfnew = pivot_top10.fillna(value=fill_values)
dfnew

# Pandas has other methods for filling the missing values including forward and backward filling. Forward filling replaces the missing values by the last valid value in the table and backward filling replaces the missing values by next valid value. These techniques are useful for sequential data such as time series and wouldn't make sense to be applied to tabular data.<br>
# To use these methods, when using `fillna` instead of passing in a value, you can pass a method. For forward filling pass in `method = "ffill"` and for backward filling pass in `method = "bfill"`.

# If you simply want to get rid of rows with missing values you can use `.dropna()`

dfnew = pivot_top10.dropna()
dfnew

# __Note:__ Notice the number of rows are much less in the table above compared to the original table.

# If we remove any row that contains missing values we might lose a significant portion of the data. Alternatively, we can only remove rows which have more than a certain number of missing values. To do so, we can set a threshold.

# remove the rows with at least 3 missing values
dfnew = pivot_top10.dropna(thresh=3)
dfnew

# Now we have more rows compared to when we removed all missing values.<br>
# Next step is to replace the missing values using the techniques discussed above.

# ## Saving Data
# After analysis and reshaping the data you might want to save the results in a file. Similar to reading files, pandas supports multiple file formats to save the tables.

pivot_top10.to_csv("final_table.csv")

# ## Pandas Plotting
# Pandas dataframes have plotting methods which help to visualise the data. The following plots are supported in pandas:
# - 'line' : line plot (default)
# - 'bar' : vertical bar plot
# - 'barh' : horizontal bar plot
# - 'hist' : histogram
# - 'box' : boxplot
# - 'kde' : Kernel Density Estimation plot
# - 'density' : same as 'kde'
# - 'area' : area plot
# - 'pie' : pie plot
# - 'scatter' : scatter plot
# - 'hexbin' : hexbin plot.
#
# You can select which plot you want to use by setting `kind` to the string for the plot.<br>
# There are a few other useful options you can set:
# - xlim, ylim: to set limits of axes
# - logx, logy, loglog: to set whether an axis should be displayed in logarithmic scale]
# - title: to set the title of the plot
# - figsize: to set the size of the plot
# <br><br>Let's try a few types of charts and graphs.

# Top 10 postcodes with largest carbon monoxide emission
pivot_top10.sort_values(by="Carbon monoxide", ascending=False)[:10].plot(
    kind="barh", y="Carbon monoxide"
)

# histogram of benzene emission
pivot_top10.plot(kind="hist", y="Benzene", bins=50)

# +
# kernel density estimation plot of benzene emission

pivot_top10.plot(kind="kde", y=["Benzene", "Toluene (methylbenzene)"], logx=True)
# -

# histogram of benzene emission in each postcode
pivot_top10.plot(kind="box", logy=True, rot=90)

pivot_top10.plot(kind="scatter", x="Toluene (methylbenzene)", y="Benzene", loglog=True)

# pie chart of emission of the substances in postcode 4008
pivot_top10.loc[4008, :].plot(kind="pie", subplots=True, figsize=(10, 10))

# We will discuss producing more advanced plots in the next notebooks where we learn about various plotting packages in python.

# ## Further reading
# - [Pandas documentation](https://pandas.pydata.org/)
# - [Pandas in 10 minutes](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)
# - https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf
# - https://www.kaggle.com/learn/pandas
# - https://www.kaggle.com/kashnitsky/topic-1-exploratory-data-analysis-with-pandas
# - https://www.youtube.com/watch?v=ZyhVh-qRZPA&list=PL-osiE80TeTsWmV9i9c58mdDCSskIFdDS
#


