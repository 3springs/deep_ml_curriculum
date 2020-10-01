# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: deep_ml_curriculum
#     language: python
#     name: deep_ml_curriculum
# ---

# # Time Series Forcasting

#
# In time series forcasting (TSF) the goal is to predict the future values using the behaviour of data in the past. We can use some of the tehniques we learned about in the last notebook. For instance, Holt-Winters methods can be used for forcasting as well as analysis.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
plt.rcParams["figure.figsize"] = [12,5]
warnings.simplefilter("ignore")

# We will load a subset of London Smart meters dataset. This dataset shows electricity consumption of 5,567 houses in London. We will only use the data for a single block. 
#
# The data shows daily consumption of each house and various statistics regarding their daily consumption. The original data is from [UK Power Networks](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households)

# Load data
df = block0 = pd.read_csv("../../data/processed/smartmeter/block_0.csv", parse_dates=['day'], index_col=['day'])[['energy_sum']]
# Get the mean over all houses, by day
df = df.groupby('day').mean().iloc[:-1]
# Rename energy to target
df = df.rename(columns={'energy_sum':'target'})
df.plot()
df

# In forcasting we try to predict the next step, therefore it is essential that we specify the frequency of data so the model knows what we mean by next step. 
#
# Pandas data frames have frequency property, which we need to set

df.index

# You can see at the bottom `freq` is set to `None`. We need to specify the the data is monthly and the dates are start of the month. So we use `freq = "MS"`.

df.index.freq = "1D"

# __Note:__ Most of the algorithms have ways of infering the frequency if it is not set. But it is always safer to set it ourselves rather than leave it for the algorithms to figure out.

# To measure whether we are doing well in our prediction or not, commonly we split the data into two parts, one for training the model and the other for evaluating the forcasting quality. In time series we train on the past and predict on the future, so the validation set needs to be in the future.  
#
# The part that is used for taining is called training set and for time series it usually is the data from the beginning up to a certain point in time. The part that is used for evaluation is may be called validation set, test set, or evaluation set. The validation set comes right after the training set, because we use the training set to understand the behaviour of data and then we want to know what is going to happen right after that.
#
#
# Let's split our data into training and validation set. Let's split in a way so that last 30% is in validation set and the rest in training set.

# +
# We are forecasting, so split into past and future
n_split = -int(len(df)*0.7)
df_train = df[:-n_split]
df_valid = df[-n_split:]

ax = df_train['target'].plot(legend=True, label="Train")
df_valid['target'].plot(ax=ax, legend=True, label="Validation")
# -

# ## Stationarity
#
# A time series is considered stationary when its properties (mean and standard deviation) does not change with time. Therefore, any time series with trend or seasonality is not stationary. An example of stationary data is white noise:

plt.figure(figsize=(12, 8))
plt.plot(range(100), np.arange(100)/50, ls=':', c='b', label='line - not stationary')
plt.plot(range(100),np.sin(np.arange(100)/5)-2, c='b', label='sin - not stationary')
plt.plot(range(100), np.zeros(100), c='r', label='zeros - stationary')
plt.plot(range(100), np.random.randn(100)+4, ls='--', c='r', label='random noise - stationary')
plt.legend()
plt.xlabel('time [days]')
plt.title('examples of non/stationary series')

# Why is random noise stationary?
# The std and mean are constant
np.random.seed(42)
random_noise = pd.Series(np.random.randn(200))
plt.plot(random_noise, label='random noise')
random_noise.rolling(30).mean().plot(label='mean')
random_noise.rolling(30).std().plot(label='std')
plt.legend()

# Sin - this is not stationary
# The std and mean are not constant
np.random.seed(42)
series_sin = pd.Series(np.sin(np.arange(200)/5))
plt.plot(series_sin, label='sin(x/5)')
series_sin.rolling(50).mean().plot(label='mean')
series_sin.rolling(50).std().plot(label='std')
plt.legend()

# While it is easy to tell if a time series is not stationary when there is a clear trend, in some cases it might be pretty difficult to decide whether a time series is stationary or not. Therefore, we use statistical tests to make a decision.
#

# __Why is it important if a time series is stationary or not?__<br>
# We know that in a stationary time series the characteristics will remain constant. This makes it easier to predict their future behaviour as we expect them to behave similarly. But when the series is not stationary we don't know how it is going to behave in the future. In reality, most of the time series we are going to work with are not stationary. But using various techniques we might be able to transform them into a stationary time series. This is exactly what we just did. We use STL to remove the trend and seasonality to get a stationary time series.

# #### Augmented Dickey-Fuller test
#
# [Augmented Dickey-Fuller test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test) (ADF) is a statistical test for stationarity. We are not going to discuss the statistical details of this test, but what matters to us is the result. 
#
# The null hpothesis of ADF is: `the series is stationary.`
#
# Let's test it on our data.
#

# +
from statsmodels.tsa.stattools import adfuller

def adf_p_value(data):
    p = adfuller(data)[1]
    
    # If p-value is lower than a threshold (commonly 0.05),
    if p<0.05:
        # it means the null hypothesis is rejected and therefore the time series is stationary.
        return f'stationary (p={p:2.2g})'
    else:
        return f'not stationary (p={p:2.2g})'


# -

adf_p_value(df["target"])

# The function returns many values, but the one that we are interested in is p-value, which the second value. If it is less than 0.05, it means time series is stationary. In this case it is far from 0.05 and that is what we expected as the data has clear trend.<br>
# Now let's turn it into a function that only return the p-value and run the test on white noise.

adf_p_value(random_noise)

# The value is very small, which suggests we can reject the null hypothesis and therefore the series is stationary.

# ## Decomposing

# What if we remove trend and seasonality from the data using STL method?

from statsmodels.tsa.seasonal import seasonal_decompose

res = seasonal_decompose(df[:100], model="mul")
res.plot()
''

# If we remove the seasonal and trend component what is left is the residuals.<br>
# The residuals might have `NaN` in it. If so, we need to remove them before performing the test.

adf_p_value(res.resid.dropna())

# The residual is stationary since the p value is lower than 0.05.

df.plot()
df.diff().plot()
df.diff(2).plot()

# Another technique to make a time series stationary is differencing. Differencing means that we calculate the difference between two consecutive points in time. Then we use the differences for forcasting.<br>
# Let's see how differencing will affect our data. Pandas has a builtin method for differencing (`.diff()`):

df.diff()

# We need to get rid of `NaN` so we can run the test.

adf_p_value(df.diff().dropna()["target"])

# As we can see p-value is below the 0.05 threshold, which means differencing helped to convert data into stationary time series. <br>
# In some cases you might need to perform differencing multiple times to reach stationary results.

adf_p_value(df.diff(2).dropna()["target"])

# ## Autocorrelation

# Another characteristics of a time series is autocorrelation. Autocorrelation is simply the correlation between the points in the time series and the points before them (sometimes called lagged values).
#
# The shaded area is the confidence threshold on the correlation using Bartlett's formula $1/\sqrt{N}$ which assumes a guassian distribution. If a correlations is below this threshold is it's likely to be a coincidence.

from statsmodels.graphics.tsaplots import plot_acf
df.plot()

plot_acf(df)
plt.xlabel('Lag (day)')
plt.ylabel('Correlation coeffecient')
''

# The points closer together in time have higher correlation compared to the points further apart. This is an expected behaviour. However, how quickly does the correlation decreases is important.

# ## Autoregressive models (AR)
# An [autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model), is a time series model which assumes a linear relationship between each point in time and its past $p$ points.
#
# $$y_t=c+\sum_{i=1}^{p}\phi_iy_{t-i}$$
# For instance a first order AR (also shown as AR(1)) can be written as:<br>
# $$y_t=c+\phi_1 y_{t-1}$$
# This model can be found in statsmodels in ar_model submodule.

# This is to avoid some warning messages from statsmodels
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from statsmodels.tsa.ar_model import AR, ARResults

# Let's try an AR model on our data. 

# +

model = AR(df_train)

# Then we train the model specifying the order of AR. Let's start by trying `1`.
trained_model = model.fit(
    maxlag=2,  
    trend='nc',
)

# Now the model is trained. We can view model's values:
print('params\n', trained_model.params)
# -

# More importantly, we can forecast using the trained model. 

# +
# specify at which time-step in the training data the model should start and at which time-step it should stop
start = len(df_train)
end = len(df_train) + len(df_valid) - 1
forecast = trained_model.predict(start, end)

fig = plt.figure()
ax = fig.gca()
df_train['target'].plot(ax=ax, legend=True, label="Train")
df_valid['target'].plot(ax=ax, legend=True, label="Actual")
forecast.plot(ax=ax, legend=True, label="Forecast")
# -

# ## Metrics

# It's not very close. But how close? We need to put a value on the goodness of the result. To do this, we can use metrics. There are various metrics which can be used here, such as root of mean squared error (RMSE), mean squered error (MSE), mean absolute error (MAE), $R^2$, and many more. Sometimes for a certain application you might need to use particular metric.<br>
#
# We will use Mean Absolute Percentage Error.
#
# $$MAPE=\frac{\lvert y_{true}-y_{pred}\rvert}{y_{true}}$$
#
# There is a package called Scikit Learn which is a commonly used for machine learning and data science. This package contains many useful functions and algorithms. One of them is the metrics submodule where various types of metrics are available. 
#
#

# +
from sklearn.metrics.regression import _check_reg_targets

def mape(y_true, y_pred, epsilon=1e-3):
    """
    Mean absolute percentage error
    
    This function already exists in newer versions of sklearn.
    
    https://scikit-learn.org/dev/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
        y_true, y_pred, 'uniform_average')
    
    # This is the important line
    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.average(mape)



# -

# That's not so good! Let's calculate mean absolute error:

mape(df_valid, forecast)

# Now let's try larger models by increasing order of AR. It looks at a longer term trend now.

model = AR(df_train)
model = model.fit(maxlag=7,  trend='nc')
start = len(df_train)
end = len(df_train) + len(df_valid) - 1
forecast = model.predict(start, end)
fig = plt.figure()
ax = fig.gca()
df_train['target'].plot(ax=ax, legend=True, label="Train")
df_valid['target'].plot(ax=ax, legend=True, label="Actual")
forecast.plot(ax=ax, legend=True, label="Forecast")
mape(df_valid, forecast)

# Note that the MAPE is lower, meaning it is a better fit

# <div class="alert alert-success">
#   <h2>Exercise</h2>
#
#   Try a few other values yourself and see if you get a better/lower result  than mape=0.4
#   
#   - try trend='nc', which makes it return to the mean.
#   - try a great lag, which gives it more parameters
#     
#     
#   Does it *look* better as well? Is MAPE capturing your intuition about a good fit?
#       
#
#   <details>
#   <summary><b>→ Hints</b></summary>
#
#   * try `model.fit(maxlag=30,  trend='c')`
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
#   ```python
#     model = AR(df_train)
#     model = model.fit(maxlag=30,  trend='nc')
#     start = len(df_train)
#     end = len(df_train) + len(df_valid) - 1
#     forecast = model.predict(start, end)
#     fig = plt.figure()
#     ax = fig.gca()
#     df_train['target'].plot(ax=ax, legend=True, label="Train")
#     df_valid['target'].plot(ax=ax, legend=True, label="Actual")
#     forecast.plot(ax=ax, legend=True, label="Forecast")
#     mape(df_valid, forecast)
#   ```
#
#   </details>
#
#   </div>
#
#



# # Prophet
# Prophet is a time series analysis and forecasting package developed by Facebook. Prophet allows you to train forecasting models with minimal need to adjust the models parameters. Prophet is particularly useful when you are dealing with data that has multiple levels of seasonality.
#
# Let's start by importing the library. The name of the package is `fbprophet`.

from fbprophet import Prophet

# +
# Load data
df = block0 = pd.read_csv("../../data/processed/smartmeter/block_0.csv", parse_dates=['day'], index_col=['day'])[['energy_sum']]
# Get the mean over all houses, by day
df = df.groupby('day').mean()
# Rename energy to target
df = df.rename(columns={'energy_sum':'target'}).iloc[:-1]

n_split = -int(len(df)*0.85)
df_train = df[:-n_split]
df_valid = df[-n_split:]

ax = df_train['target'].plot(legend=True, label="Train")
df_valid['target'].plot(ax=ax, legend=True, label="Validation")

# df.plot()
# -



# Prophet needs the input data to be in a very specific format. The data needs to have a column containing daily dates called `"ds"`, and a column containing values named `"y"`. So we create a new data frame and use the required column names.

# +
df_trainp = pd.DataFrame({"ds": df_train.index, "y": df_train["target"]}).reset_index(drop=True)
df_trainp

df_validp = pd.DataFrame({"ds": df_valid.index, "y": df_valid["target"]}).reset_index(drop=True)
df_validp
# -

# Now the data is ready. We need to create a Prophet model and train it on the data.



# %%time
model = Prophet(holidays_prior_scale=0.01)
model.fit(df_trainp)

# And that's it! The model is trained and ready to be used.<br>
# Let's forecast the next year using the model. To forecast using Prophet we need to first create an empty dataframe for future values. This data frame contains the future dates. Then we feed this dataframe to `.predict()` and will get the forecasted values.



future = model.make_future_dataframe(periods=365)
future.head()

# __Note:__ as you can see this data frame has only future dates.

forecast = model.predict(future)
forecast.head()

# The result contains various components of the time series. The forecasted values can be found on `yhat` column. It is difficult to see how model has performed, so let's plot the results. We can do that using Prophets built-in plot function.

fig = model.plot(forecast)
fig.gca().plot(df_validp.ds, df_validp['y'], 'k.', c='r', label='validation')
plt.legend()
''

# As you can see at some periods the predictions are poor and at some points they are pretty close. Let's have a closer look at the future.

fig = model.plot(forecast)
fig.gca().plot(df_validp.ds, df_validp['y'], 'k.', c='r', label='validation')
plt.xlim(pd.to_datetime(["2013-06-15", "2013-08-15"]))
plt.ylim([10, 24])

# The model has found annual and weekly seasonalities. We can have closer look at these components using `.plot_components()`

model.plot_components(forecast)
1

# Now we can see which days of the week are associated with more energy consumption (it's not suprising to see Saturday and Sunday) and also how time of the year affects the energy consumption.

# ## Cross_validation

# We created a model and forecasted the future. But we still don't know how good the model is. 
#
# So like before we need a training and a validation set. We train a model on a training set, and then measure the accuracy of its prediction on validation set using metrics.
#
#
# One issue with this approach is that even when we get a value for prediction accuracy of a model, how do we know this value is reliable. Let's say we are comparing two models and mean absolute error for model A is 0.5 and for model B is 0.45. How do we know that B is better than A and it didn't just get lucky over this data set? 
#
# One way to ensure which one is better is by comparing them over multiple sections of data sets. This approach is called `cross validation`. In Prophet, we start by training the model over the data from the beginning up to a certain point (cut-off point) and then predict for a few time steps (Horizon). Then we move the cut-off point by a certain period and repeat the process. We can then calculate the metrics for each model over multiple sections of the data and have a better comparison at the end.
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/b/b5/K-fold_cross_validation_EN.svg"/>

# You need to specify the following inputs:
# - initial: The initial length of training set.
# - period: How much the cut-off point is moved after each training process.
# - horizon: Length of forecasting period for which the metrics are calculcated.
#

# +
from fbprophet.diagnostics import cross_validation

# Cross validation
cv = cross_validation(model, initial="365 days", period="90 days", horizon="30 days")
cv.head()
# -



# The cross validation data frame shows the forecasted value (yhat) and its confidence range (yhat_upper and yhat_lower). We can use `performance_metrics` function to calculate the metrics.

# +
from fbprophet.diagnostics import performance_metrics

perf = performance_metrics(cv)
perf.index = pd.Index(perf.horizon.dt.days, name='days')
perf
# -
# The dataframe above has multiple metrics for model's predictions.
#

# <font color = red>__Note:__ In some versions of Prophet all the result is aggregated based on how far forecasted point is from cut-off point. If this is not the case, then you will see horizon column has repeated values (for instance multiple "10 days" entries) and you will need to use groupby.</font>

# +
# uncomment and run if performance metrics are not aggregated based on horizon days

# perf = perf.groupby('horizon').mean()
# perf
# -

# Before running the next cell look at the performance data frame and find the first and last horizon days and enter it in the next cell as `start` and `end`.

perf[['mape']][:-1].plot(ylim=[0, 0.3])
plt.title("Mean Absolute Percent Error of forecasts")

# This plot shows the further we are from the cut-off point the larger the error is, which is what we expect. Now, let's compare this model with another one.<br>
#

# ## Holidays
#
# Prophet has the ability to include the effect of holidays on the time series as well. Let's see whether adding public holidays to the model will make it any better.

holiday_df = pd.read_csv(
    "../../data/processed/smartmeter/uk_bank_holidays.csv",
    names=("ds", "holiday"),
    header=0,
)
holiday_df.head()



# +
model2 = Prophet(holidays=holiday_df)
model2.fit(df_trainp)

# Cross validation
cv2 = cross_validation(model2, initial="365 days", period="90 days", horizon="30 days")
perf2 = performance_metrics(cv2)
perf2.index = pd.Index(perf2.horizon.dt.days, name='days')
perf2

# +
# uncomment and run if performance metrics are not aggregated based on horizon days

# perf2 = perf2.groupby('horizon').mean()
# perf2
# -

# Now let's compare the models.

ax=plt.gca()
perf['mape'][:-1].plot(ax=ax, label="+ holidays")
perf2['mape'][:-1].plot(ax=ax, label="- holidays")
plt.title("Mean Absolute Percent Error of forecasts")
plt.ylabel("mape")
plt.legend()

# It seems adding holidays slightly lowered the error for the first couple of weeks.

# We can separately plot the models including the errors for all the horizons.

# +
from fbprophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(cv, metric="mape")
plt.ylim(0, 1)
plt.xlim(0, 27)
# -

# ## Trends

# One interesting feature of Prophet is that it can identify when the trend of the data is changing. We can add these change points to the plot as well.

# +
from fbprophet.plot import add_changepoints_to_plot

model = Prophet()
model.fit(df_trainp)
future = model.make_future_dataframe(periods=len(df_validp))
forecast = model.predict(future)
fig = model.plot(forecast)
ax = fig.gca()
a = add_changepoints_to_plot(ax, model, forecast)
ax.plot(df_validp.ds, df_validp['y'], 'k.', c='r', label='validation')
# -



# We can change the sensitivity of the model to the changes by setting `changepoint_prior_scale`.

model = Prophet(
    changepoint_range=0.90,
    changepoint_prior_scale=0.2,
)
model.fit(df_trainp)
future = model.make_future_dataframe(periods=len(df_validp))
forecast = model.predict(future)
fig = model.plot(forecast)
ax = fig.gca()
a = add_changepoints_to_plot(ax, model, forecast)
ax.plot(df_validp.ds, df_validp['y'], 'k.', c='r', label='validation')





# Prophet has many other parameters you can set to improve your model, including seasonality, growth type, etc. You can find more information about Facebook Prophet [here](https://facebook.github.io/prophet/docs/diagnostics.html).

# # Exercise

# Now that we have learned about various time series forecasting techniques, try to apply some of these techniques to another block of houses from electricity usage.

#  <div class="alert alert-success">
#   <h2>Exercise</h2>
#
#   Now that we have learned about various time series forecasting techniques, try to apply some of these techniques to another block of houses from electricity usage.
#
#   ```python
#     # Load data
#     df = block1 = pd.read_csv("../../data/processed/smartmeter/block_1.csv", parse_dates=['day'], index_col=['day'])[['energy_sum']]
#     # Get the mean over all houses, by day
#     df = df.groupby('day').mean()
#     # Rename energy to target
#     df = df.rename(columns={'energy_sum':'target'}).iloc[:-1]
#
#     n_split = -int(len(df)*0.85)
#     df_train = df[:-n_split]
#     df_valid = df[-n_split:]
#
#     df_trainp = pd.DataFrame({"ds": df_train.index, "y": df_train["target"]}).reset_index(drop=True)
#     df_validp = pd.DataFrame({"ds": df_valid.index, "y": df_valid["target"]}).reset_index(drop=True)
#     
#     # COPY PREVIOUS CELL HERE (And change parameters)
#   ```
#       
#
#   <details>
#   <summary><b>→ Hints</b></summary>
#
#   * Copy the cell above, and enter the new dataframe
#   * Perhaps try `Prophet(    changepoint_range=0.90, changepoint_prior_scale=0.2, holidays=holiday_df,)`
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
#   ```python
#     # Load data
#     df = block1 = pd.read_csv("../../data/processed/smartmeter/block_1.csv", parse_dates=['day'], index_col=['day'])[['energy_sum']]
#     # Get the mean over all houses, by day
#     df = df.groupby('day').mean()
#     # Rename energy to target
#     df = df.rename(columns={'energy_sum':'target'}).iloc[:-1]
#
#     n_split = -int(len(df)*0.85)
#     df_train = df[:-n_split]
#     df_valid = df[-n_split:]
#
#     df_trainp = pd.DataFrame({"ds": df_train.index, "y": df_train["target"]}).reset_index(drop=True)
#     df_validp = pd.DataFrame({"ds": df_valid.index, "y": df_valid["target"]}).reset_index(drop=True)
#
#     # help(Prophet)
#     model = Prophet(
#         changepoint_range=0.90,
#         changepoint_prior_scale=0.2,
#         holidays=holiday_df,
#     )
#     model.fit(df_trainp)
#     future = model.make_future_dataframe(periods=len(df_validp))
#     forecast = model.predict(future)
#     fig = model.plot(forecast)
#     ax = fig.gca()
#     a = add_changepoints_to_plot(ax, model, forecast)
#     ax.plot(df_validp.ds, df_validp['y'], 'k.', c='r', label='validation')
#   ```
#
#   </details>
#
#   </div>

# +
# Load data
df = block1 = pd.read_csv("../../data/processed/smartmeter/block_1.csv", parse_dates=['day'], index_col=['day'])[['energy_sum']]
# Get the mean over all houses, by day
df = df.groupby('day').mean()
# Rename energy to target
df = df.rename(columns={'energy_sum':'target'}).iloc[:-1]

n_split = -int(len(df)*0.85)
df_train = df[:-n_split]
df_valid = df[-n_split:]

df_trainp = pd.DataFrame({"ds": df_train.index, "y": df_train["target"]}).reset_index(drop=True)
df_validp = pd.DataFrame({"ds": df_valid.index, "y": df_valid["target"]}).reset_index(drop=True)

# COPY PREVIOUS CELL HERE (And change parameters)
# -

# # (Advanced) Custom Seasonality
#
# This library is made by facebook for tracking user trends. That means it is set up for growing tends to do with humans, with holidays and weekly seasonality. What if we have data that has a differen't seasonality?
#
# Here we use current speed from the [IMOS - Australian National Mooring Network (ANMN) Facility - Current velocity time-series](https://catalogue-imos.aodn.org.au/geonetwork/srv/api/records/ae86e2f5-eaaf-459e-a405-e654d85adb9c). We will use tidal periods related to the Sun and Moon instead of human calender periods related to Weeks and Holidays.

# +
# from https://catalogue-imos.aodn.org.au/geonetwork/srv/api/records/ae86e2f5-eaaf-459e-a405-e654d85adb9c
import xarray as xr
xd = xr.open_dataset("../../data/processed/IMOS_ANMN/IMOS_ANMN-WA_AETVZ_20111221T060300Z_WATR20_FV01_WATR20-1112-Continental-194_END-20120704T050500Z_C-20200916T043212Z.nc")
name='CSPD'
df = xd.isel(HEIGHT_ABOVE_SENSOR=0)['CSPD'].isel(TIME=slice(0, -1000)).to_dataframe()[['CSPD']]

# Take the log, and smooth it by resampling to 4 hours
df['CSPD'] = np.log(df['CSPD'])
df = df.resample('4H').mean()

# Format for prophet
df = pd.DataFrame({"ds": df.index, "y": df['CSPD']})

# Split
n_split = -int(len(df)*0.7)
df_trainp = df[:-n_split]
df_validp = df[-n_split:]

ax = df_trainp['y'].plot(legend=True, label="Train")
df_validp['y'].plot(ax=ax, legend=True, label="Validation")
plt.ylabel('Current Speed')

# +
# %%time
# First let's try it with the default calender/holiday seasonalities
model = Prophet()

model.fit(df_trainp)

forecast = model.predict(df_validp)
forecast.index = forecast.ds

fig = model.plot(forecast)
a = add_changepoints_to_plot(plt.gca(), model, forecast)
fig.gca().plot(df_validp.index, df_validp['y'], 'k.', c='r', label='validation')
plt.show()
''
# -

# %%time
# Cross validation
cv = cross_validation(model, horizon="7 days", period="4 days", initial="60 days", parallel='threads')
perf = performance_metrics(cv)
perf.index = pd.Index(perf.horizon.dt.days, name='days')
print('mape', perf.mape.mean())
# perf.mape.plot()
# perf

model.plot_components(forecast)
1

# This is tidal data, and the default (daily, weeklly, yearly) seasonalities don't capture the dominant monthly seasonality in the tides. Lets add tidal frequencies and see if it does better.
#
# Also not that we have made growth flat, since tides tend to return to the mean.

# +
# %%time

model = Prophet(
    growth='flat', # Recent addition https://github.com/facebook/prophet/pull/1466
    
    # Disable default seasons
    yearly_seasonality=False,
    holidays=None,
    daily_seasonality=False,
    weekly_seasonality=False,
    holidays_prior_scale=0.001,
)

# Add periods from the theory of tides https://en.wikipedia.org/wiki/Theory_of_tides (additive)
# Period is in days
# Fourier order is how many fourier functions can be used, higher is more complex and unstable


# Short
# model.add_seasonality(name='M4', period=6.21/24, fourier_order=1)
# model.add_seasonality(name='M6', period=4.14/24, fourier_order=1)
# model.add_seasonality(name='M6', period=8.17/24, fourier_order=1)

# Semi-diurnal
model.add_seasonality(name='M2', period=12.4206012/24, fourier_order=1)
model.add_seasonality(name='S2', period=12/24, fourier_order=1)
# model.add_seasonality(name='K2', period=12.65834751/24, fourier_order=1)

# diurnal
model.add_seasonality(name='K1', period=23.93447213/24, fourier_order=2)
# model.add_seasonality(name='O1', period=25.81933871/24, fourier_order=2)

# Monthly and higher
model.add_seasonality(name='Mm', period=27.554631896, fourier_order=2)
# model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
# model.add_seasonality(name='Ssa', period=182.628180208, fourier_order=1)
# model.add_seasonality(name='Sa', period=365.256360417, fourier_order=1)

model.fit(df_trainp)

forecast = model.predict(df_validp)
forecast.index = forecast.ds

fig = model.plot(forecast)
fig.gca().plot(df_validp.index, df_validp['y'], 'k.', c='r', label='validation')
a = add_changepoints_to_plot(plt.gca(), model, forecast)
plt.show()
''
# -

# Cross validation
cv = cross_validation(model, horizon="7 days", period="4 days", initial="60 days", parallel='threads')
perf = performance_metrics(cv)
perf.index = pd.Index(perf.horizon.dt.days, name='days')
print('mape', perf.mape.mean())
# perf.mape.plot()
# perf

model.plot_components(forecast)
1

# # Further Reading
# - [Introduction to the Fundamentals of Time Series Data and Analysis](https://www.aptech.com/blog/introduction-to-the-fundamentals-of-time-series-data-and-analysis/)
# - [The Complete Guide to Time Series Analysis and Forecasting](https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775)
# - [Generate Quick and Accurate Time Series Forecasts using Facebook’s Prophet](https://www.analyticsvidhya.com/blog/2018/05/generate-accurate-forecasts-facebook-prophet-python-r/#:~:text=Prophet%20is%20an%20open%20source,of%20custom%20seasonality%20and%20holidays!)
# - [Facebook Prophet Documentation](https://facebook.github.io/prophet/docs/quick_start.html#python-api)
#




