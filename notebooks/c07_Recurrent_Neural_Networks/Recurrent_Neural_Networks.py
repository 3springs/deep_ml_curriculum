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

# + [markdown] colab_type="text" id="iALm8shtXMVK"
# # Recurrent Neural Networks
#
# All the models we have discussed so far were looking at the inputs as isolated instances. In image classification we determine the category of and image, in semantic segmentation we produce a semantic map from the input. But there are many cases were datapoints are not isolated instances and have connection to each other. Sequential data are the type of data where each instance is related to the instances came before. A good example for this type of data is time series data. At each point in time to the value of the time series depends on the value of the prior points. Recurrent Neural Networks (RNN) are a class of networks which deal with sequential data. There are many variants of Recurrent Neural Networks, including:
#
# - Simple Recurrect Neural Networks (Simple RNN - or often just called RNN)
# - Long Short-Term Memory (LSTM)
# - Gated Recurrent Unit (GRU)
#
# In this notebook we will discuss LSTM; however, the general logic behind all these methods are the same. They only differ in the way they handel information internally. 
# <img src='./images/RNN.png'>
# <div style="font-size:70%">Recurrent Neural Networks Architecture - Credit to <a href='https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180944'>Bao et al.</a></div>
# At each step a set of values are stored as state of the model ($s$). The state is updated at each step and based on the value of the state the output of the model is calculated by passing state through the output layer ($V$). Also, the model takes an input $x_t$ and passes it through an input layer ($U$), The output of this layer is combined with the state of previous step and creates the new state ($s_t$). What is important here is that the input and output layers are the same for all the sequences of data. What makes the output different at each step are the input values and state of the model. <br>
# Various architectures handel updating the state value differently and this is the main difference between Simple RNN, LSTM, and GRU. 
# -

# ## Single Variate Time Series
#  Recurrent Neural Networks can be quite useful for time series forecasting due to their ability to understand sequential nature of data. For the start we can see how an LSTM can forcast a time series with only one variable.

# ### Predicting Sine Function
# In the first example we will use an LSTM model to predict the value of sine function. Since the sine function is easy to calculate we can easily evaluate the performance of the model. 

# + colab={} colab_type="code" id="eXmrb7haVf1f"
import torch
from torch import nn, optim
from torch import functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# -

# Let's create some data for training:

# + colab={} colab_type="code" id="HRbaIjQaVxRF"
t = np.linspace(0, 100, 1000)
sine = np.sin(t)

# + colab={"base_uri": "https://localhost:8080/", "height": 282} colab_type="code" id="TkCUDqPfWF17" outputId="6a0fb3c3-4bfd-4245-bf31-4ca70b914330"
plt.plot(t, sine)


# + [markdown] colab_type="text" id="kSFDSn41i84G"
# #### Creating the model
# In an LSTM the model takes one data point at the time and produces an output. This output is also called the hidden state. We can use the hidden state to calculate the next value in the sequence. The hidden state helps the model to remember what happened before and not just look at each point as an isolated instance. <br>
# Pytorch LSTM component can needs a few inputs:
# - __input_size:__ The size of the input data. In this example since we are dealing with a single variate data we pass in one value at a time, therefore, the input size is one.
# - __hidden_size:__ This represents how many values should be stored as the hidden state in the model. Larger values allows model to understand more complex patterns in the data.
# - __num_layers:__ This value refers to the number of layers in the LSTM block. Similar to what we saw in tabular data, having multiple layers allows the model to work with more complex patterns.
#
# When we are creating our model we add a linear layer as well which will be our output layer. The output layer converts the hidden state to output value. Depending on what we want the output to be we can choose the size of output as well. In this example, we are trying to find the next value in the series, therefore, the model only returns a single value.
#
# TODO, Mike add fig
# TODO Make a coherent story
# - there are multiple ways to frame a timeseries prediction problem
#     - windowed - fixed content length
#     - recursive - promises to let the network decide on context length
# - couldn't get recursive timeseries prediction to work with just a linear layer. The problem is exploding and vanishing gradients.
#     - LSTM gives it ability to forget and remember
# - LSTM's have got SOTA result on many datasets. they are quite robust, and win a lot of competitions.

# + colab={} colab_type="code" id="z3ZJWs-CWJTo"
# TODO use just seq(LSTM, linear)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = next(iter(self.parameters())).device
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_n = h_n.view(-1, self.num_layers, self.hidden_size)[:, -1]
        out = self.linear(h_n)
        return out


# +
# lstm = nn.LSTM(
#             input_size=3,
#             hidden_size=44,
#             num_layers=2,
#             batch_first=True,
# )
# # lstm?
# -

# ### Preparing Data
# The data for an LSTM model is a bit different from what we have seen so far. Since this model is supposed to look at a few data points to predict the next one, data needs to be broken down to smaller sections. At each step we pass a few data points into the model and the model will forecast the next. The longer this sequence of input is the better the model will be, however, this also means we require larger memory and longer training time.<br>
# The function below will create the inputs and targets for the model.
#

def create_seq_data(series, seq_length):
    sequence = []
    target = []
    for i in range(len(series) - 1 - seq_length):
        sequence.append(series[i : i + seq_length])
        target.append(series[i + 1])
    return np.array(sequence), np.array(target)


# + [markdown] colab_type="text" id="QlDuDJZ2m4-a"
# Let's split the data into a training and test set. Note that here `x` contains sets of 5 consecutive numbers and `y` contains the respective number that comes after each serie.

# + colab={} colab_type="code" id="7Nun30E1mlme"
seq_length = 5
x, y = create_seq_data(sine, seq_length)

xtrain = x[:600, :]
ytrain = y[:600]

xtest = x[600:, :]
ytest = y[600:]

# + [markdown] colab_type="text" id="LyIDJnNzoajf"
# Let's print xtrain to better understand what it contains.

# + colab={"base_uri": "https://localhost:8080/", "height": 136} colab_type="code" id="D_6Bp622ms4q" outputId="e2463f15-d627-4a58-c63f-3b2ca647f27d"
print(xtrain)
# -

# Now we can create an instance of the model with input size of one (single variate series), one layer LSTM with hidden state size of 50.

# + colab={} colab_type="code" id="jkGyTt5hZwg6"
model = LSTM(input_size=1, hidden_size=50, num_layers=1)
# -

# Also we need to choose the optimiser, and loss function. Let's use mean square error loss and ADAM optimiser.

loss_func = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# #### Training the model
# Now that all the components are ready we can train the model.

# + colab={"base_uri": "https://localhost:8080/", "height": 236, "referenced_widgets": ["2249f632765e44fb873c5b52e38952b3", "ea24cf68548440d6bb27527e1e8b231a", "2c444039590c49d7b0ba75e37b9396a2", "779b8d08fe4d424296bd06322a8c055b", "2708bb5b48024a0a97470b123004e914", "239a06e64dee49a0bc6a64ae48782dfa", "ec7d5619230044ab8149273a33c7d1db", "0de1f271c85a490db7b212f29a2e98db"]} colab_type="code" id="Au3Ww4gjaJvd" outputId="f9c973f3-d710-4965-afa4-91bc07b52ad2"
epochs = 500
model.train()
for epoch in tqdm(range(epochs)):
    input_values = torch.Tensor(xtrain).unsqueeze(-1)
    optimizer.zero_grad()
    preds = model(input_values)
    loss = loss_func(preds, torch.Tensor(ytrain).unsqueeze(-1))
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"#{epoch} Loss = {loss.item():.5f}")
# -

# Note that when using `xtrain` and `ytrain` we first converted them to tensors, then used `unsqueeze(-1)`. `unsqueeze(-1)` adds a dimension to the end of the tensor. Why did we do that? The reason is to specify that the model is dealing with only one value.

print("Normal shape:")
print(torch.Tensor(xtrain).shape)
print("\nUnsqueezed(-1) shape")
print(torch.Tensor(xtrain).unsqueeze(-1).shape)

# Prior to unsqueezing the shape is the number of datapoints by the length of sequences. But after unsqueezing we have `1` as the last dimension which specifies that we are passing in one value. When we unsqueeze `y` we are specifying that the output is one value.

# #### Forecasting
# Now let's look at the results and see how good the model is at forecasting.

# + colab={} colab_type="code" id="RnlPIriDznd-"
model_input = torch.Tensor(xtest).unsqueeze(-1)
pred_test = model(model_input)

# + colab={"base_uri": "https://localhost:8080/", "height": 333} colab_type="code" id="RM-J4ZxZ0EJX" outputId="4fc5a147-3343-4f0f-a96f-02f4f8bfd6c5"
plt.figure(figsize=(15, 5))
plt.plot(to_numpy(pred_test), "x", label="Predictions")
plt.plot(ytest, label="Target")
plt.legend()

# + colab={} colab_type="code" id="R2Nz6vSbqYDE"


# + [markdown] colab_type="text" id="ZZMPJEkXq35V"
# ### Predicting Customers
# Let's start working with a real time series data with seasonality and trend. Unlike traditional techniques we don't need to tell the model about the trend and seasonality. LSTM will be able to recognise the patterns on its own.
# -

# #### Preparing Data

# + colab={} colab_type="code" id="0-GS6-GLrBGT"
import pandas as pd

# + colab={} colab_type="code" id="nWUjnJrOrVsr"
df = pd.read_csv("../../data/processed/Customers.csv")

# + colab={"base_uri": "https://localhost:8080/", "height": 282} colab_type="code" id="h8wQjjWtr1xk" outputId="a4605e4e-1d61-448b-9d4e-7ed5c3c7b9c8"
df["Customers"].plot()
# -

# To prepare the data for the model it is highly recommended to normalise the data. There are different ways to do that. You can use standard scaling or min-max scaling, etc. The advantage of normalisation is that the model doesn't need to deal with extremely large or extremely small values, which makes it more likely the model is able to forecast based on the pattern. Here, we will use `MinMaxScaler` from Scikit Learn package.

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="Mh85UumUrZiL" outputId="afbd5dae-8497-473c-8c09-16d3b3b2ef28"
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data = scaler.fit_transform(df[["Customers"]])
# -

# Now we need to decide how many values to use for predicting. Here we are going to use 12, since we know data is monthly and has annual seasonality. Therefore seeing the past 12 months allows the model the see a full perioid. We can also choose larger or smaller values and see how it would affect the results.

seq_length = 12
x, y = create_seq_data(data, seq_length)
x.shape, y.shape

# Note that the shape of `x` and `y` are now suitable for the model and we won't need to unsqueeze them anymore.

# + colab={} colab_type="code" id="4cluCEgXsCJI"
xtrain = x[:100, :, :]
ytrain = y[:100, :]
xtest = x[100:, :, :]
ytest = y[100:, :]
# -

# #### Training model
# We are going to use the same class as the last example so we only need to create a new instance of the same model.

# + colab={} colab_type="code" id="mSTzk6EKsRcY"
model = LSTM(1, 50, 1)
loss_func = torch.nn.MSELoss(reduction="mean")  # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# + colab={} colab_type="code" id="g_7_hUjKslvF"
def train_model(model, x, y, epochs=1):
    model.train()
    for epoch in tqdm(range(epochs)):
        input_values = torch.Tensor(x)
        optimizer.zero_grad()
        preds = model(input_values)
        loss = loss_func(preds, torch.Tensor(y))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"#{epoch+1} Loss = {loss.item():.5f}")


# + colab={"base_uri": "https://localhost:8080/", "height": 236, "referenced_widgets": ["c1048481b4694b12ab408fcb1ddb8b31", "c17715512f7c44ca9a1126844f832a41", "bf8163e5e5d640dc94e74413ce3137b5", "d6d149b77d024d8a81f0df42ee176200", "a41a857b0bcf4d16b95599f019f125bf", "4e2682af756042e5808bb471c318a154", "70ccc0ca7efc48599f9514c6201b2073", "f3806fac15e545f597473d41d3708200"]} colab_type="code" id="_RqKioH6s-36" outputId="f9458fd4-aabd-4ae0-b368-61c4c4db9b93"
train_model(model, xtrain, ytrain, 500)


# -

# #### Results
# Let's check out how model performs when forecasting.

# + colab={} colab_type="code" id="KCXG6wFltXoU"
def test_model(model, x, y):
    model.eval()
    model_input = torch.Tensor(x)
    pred_test = model(model_input)
    plt.figure(figsize=(15, 5))
    plt.plot(tp_numpy(pred_test), ":x", label="Predictions")
    plt.plot(y, label="Target")
    plt.legend()


# + colab={"base_uri": "https://localhost:8080/", "height": 320} colab_type="code" id="iZWmTwALwZ7K" outputId="c4858e54-450e-4191-e2c5-8abccf06d915"
test_model(model, xtrain, ytrain)

# + colab={"base_uri": "https://localhost:8080/", "height": 323} colab_type="code" id="XsRc9J3otmov" outputId="d0c53dc0-3001-4934-e23d-fdee3018cbe4"
test_model(model, xtest, ytest)
# -

# The predictions are not too bad but can certainly be improved. In most cases the model seems to underestimate the values. There are a number of techniques we can use to improve the results. One would be removing the trend from the data. By removing the trend the remaining values will be oscillating around zero and the pattern is much easier for the model to predict.<br>
# Another way to improve the model is by changing the size of the model. This can be very effective as it allows the model to handle more complex patterns.

# #### Exercise 1
# As an exercise try to improve the performance of the model:
# 1. Increase the sequence length from 12 to 18
# 2. Increase the hidden size of the model from 50 to 100

# +
# Code Here
# -

# ## Multivariate Time Series
# Now that we know how to train a model using LSTM let's start working on more difficult problems. In multivariate time series, as the name suggests, we have to deal with multiple values at a time. We need to find out how these values affect each other and forecast the next step. There are many variations to this problem. In this example we will try to forecast the value of a set of variables in the next time step. The goal is to forecast Temperature, Pressure, Relative Humidity, Wind Speed, and Dew Point, using their previous and current values.

# #### Preparing Data

# + colab={} colab_type="code" id="PJ_GLXwSu0-e"
import pandas as pd

# + colab={} colab_type="code" id="MWNgM4Keytv7"
df = pd.read_csv("../../data/processed/Generated/energy_weather.csv")

# + colab={"base_uri": "https://localhost:8080/", "height": 394} colab_type="code" id="qKWe4Yug0ImY" outputId="d5624f3d-a335-4709-be65-5556ebb4ee09"
df.head()
# -

# Select the columns we need.

# + colab={} colab_type="code" id="ijMk3dbS0J4c"
data = df[["T_out", "P_mbar", "RelH_out", "Windspeed", "Tdew"]]
# -

# This time we will use standard scaling to normalise the data.

# + colab={} colab_type="code" id="KZUjEnoZ1KHF"
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

normalised = scaler.fit_transform(data)

# + colab={"base_uri": "https://localhost:8080/", "height": 265} colab_type="code" id="YkHiIp3y2Pmc" outputId="2f029d17-b849-4530-bdfb-06f73742a305"
plt.plot(normalised, alpha=0.5)


# -

# Similar to single variate time series, we need to create data in a form that each instance of data has the values for the past few time steps.

# + colab={} colab_type="code" id="UVSyCBig2Y8I"
def create_multi_seq_data(series, seq_length):
    sequence = []
    target = []
    for i in range(len(series) - 1 - seq_length):
        sequence.append(series[i : i + seq_length, :])
        target.append(series[i + 1, :])
    return np.array(sequence), np.array(target)


# -

# Let's use 2 hours history for prediction. Therefore, the sequence length would be 12 (data points are recorded every 10 minutes).

# + colab={} colab_type="code" id="c7ruQg-p2tOf"
seq_length = 12
x, y = create_multi_seq_data(normalised, seq_length)

# We take 90% of data for training and the rest for testing
cutoff = int(len(x) * 0.9)

xtrain = x[:cutoff, :, :]
ytrain = y[:cutoff, :]

xtest = x[cutoff:, :, :]
ytest = y[cutoff:, :]

# + [markdown] colab={} colab_type="code" id="Ow4rkOgH296J"
# #### Training model
# We can still use the same class of model. We only need to set the size of input and output.

# + colab={} colab_type="code" id="4Yf3ORgo5P5V"
model = LSTM(5, 100, 1, 5)
loss_func = torch.nn.MSELoss(reduction="mean")  # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# + colab={} colab_type="code" id="udJEFkEL5wuL"
def train_model(model, x, y, epochs=1):
    model.train()
    for epoch in tqdm(range(epochs)):
        input_values = torch.Tensor(x)
        optimizer.zero_grad()
        preds = model(input_values)
        loss = loss_func(preds, torch.Tensor(y))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"#{epoch+1} Loss = {loss.item():.5f}", end="\r", flush=True)


# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["a37da43322f54d0cb3a587c85f87f146", "2e9111de005e458faa5dadf8f4757ce0", "1bb877b53ce64c5c89a0e0fddf99510f", "967cb446a5ee4a5a9f592104126b6aeb", "9113c8e51cb44ccba8f7d76dcaf1f2b3", "8f31d4f3e4f343c2b45596c7a8acd798", "996b5d7e77e14820ac4694ce102f3d2d", "3c46c1b220494ef4830e6c1f6c75f2d3"]} colab_type="code" id="oSOPC4SX6KKF" outputId="a6ea4543-74d7-40ce-abaa-d217ae3b70c6"
train_model(model, xtrain, ytrain, 100)


# -

# Note that in the examples so far we didn't print the loss value for the test set. Generally, it is recommended to uses loss value of test set to check if the model is overfitting or not.

# #### Results
# If the loss is low enough we can print the results and see how the model is doing in terms of forecasting.

# + colab={} colab_type="code" id="tC2aNxOB7iRy"
def test_model(model, x, y, plot_results=False):
    model.eval()
    model_input = torch.Tensor(x)
    pred_test = to_numpy(model(model_input))
    mse = ((pred_test - y) ** 2).mean()
    print(f"MSE = {mse:.5f}")
    if plot_results:
        plt.figure(figsize=(15, 5))
        plt.plot(pred_test, ":x", label="Predictions")
        plt.plot(y, label="Target")
        plt.legend()
    return pred_test


# + colab={"base_uri": "https://localhost:8080/", "height": 337} colab_type="code" id="hGV437-s80Pt" outputId="35c3989d-1265-4d6d-ea5b-c1a4dec541f8"
preds = test_model(model, xtest, ytest, True)

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="faZk8mPp90pf" outputId="0c0e3cfb-9ec6-43b2-da82-459b5c16cec9"
for i in range(5):
    plt.figure(figsize=(15, 4))
    plt.plot(ytest[:, i], label="Target")
    plt.plot(preds[:, i], label="Prediction")
    plt.legend()

# + [markdown] colab_type="text" id="vaL6j3pkCen3"
# ## Classification
# So far we tried single and multivariate time series using LSTM. But LSTMs are very flexible and useful for wide range of problems. For instance, we can have multi input and single output, or use a series to predict an entire different series. We can also use LSTM for classification of time series. What makes LSTM (or other types of Recurrecnt Neural Networks) interesting is that they are not just useful for time series. They can be used for any data that has sequence of values. For instance, they can be used for text prediction. If we map every word to a number then you can turn a text into a series of numbers then LSTM can be used for predicting the next word.
#

# + [markdown] colab_type="text" id="kzlqXAj4EIBN"
# In this example we are going to look at well logs which are sequential data as well.

# + colab={"base_uri": "https://localhost:8080/", "height": 255} colab_type="code" id="uNl846nE-jjq" outputId="de7b4197-6a3f-4e88-e07e-2463adba90d0"
import pandas as pd

df = pd.read_parquet(
    "../../data/processed/geolink_norge_dataset/geolink_norge_well_logs_train.parquet"
).set_index(["Well", "DEPT"])
df['Well'] = df.index.get_level_values(0)
df['DEPT'] = df.index.get_level_values(1)
df.head()
# -

# We  will stick to a gorup of long wells
df=df[df['Well'].str.startswith('35')]

df['LITHOLOGY_GEOLINK']

# +
# Remove unused categories
df['LITHOLOGY_GEOLINK'] = df['LITHOLOGY_GEOLINK'].values.remove_unused_categories()

# sort categories (leads to nicer histograms)
i = df['LITHOLOGY_GEOLINK'].values
litho_sorted = i.value_counts().sort_values(ascending=True).index
df['LITHOLOGY_GEOLINK'] = i.reorder_categories(litho_sorted, ordered=True)

df['LITHOLOGY_GEOLINK'] 
# -

# Add a well int, so the model will know what well we use
well_index = df.index.get_level_values(0)
well_int = well_index.rename_categories(range(len(well_index.categories))).astype(int)
df['Well_int']= well_int
df[['Well_int']]

# + colab={"base_uri": "https://localhost:8080/", "height": 343} colab_type="code" id="v4wy8aXrFkoK" outputId="7fa90839-9af9-4d7f-e427-e1b214740312"
# Get a list of wells, ordered by frequency
well_counts = df['Well'].value_counts()
wells = list(well_counts.index)
well_counts
# -

# Select the N longest well logs
n_wells = 10
selected_wells = wells[:n_wells]
df = df.loc[selected_wells]
df

df['LITH_ABV'] = df["LITHOLOGY_GEOLINK"].shift().fillna('Shaly Silt')
df['LITH_ABV_INT'] = df['LITH_ABV'].values.codes
df[['LITHOLOGY_GEOLINK', 'LITH_ABV']]



# +
# # We will train on measurements above Xkm depth, and test on deeper ones
# d = df['DEPT']
# test_depth = d.mean() + d.std()/2
# print(test_depth)
# df_train = df[df['DEPT']<test_depth]
# df_test = df[df['DEPT']>test_depth]
# df_train.shape, df_test.shape

# +
# We will train on measurements above Xkm depth, and test on deeper ones
# d = df['DEPT']
# m = d.groupby(level=0).mean().dropna()
# s = d.groupby(level=0).std().dropna()
# test_depth = m+s/2
# test_depth

def get_test(x):
    d = x['DEPT']
    thresh = np.round(d.mean() + d.std() / 2)
    x['thresh'] = thresh
    return x[d>thresh]

def get_train(x):
    d = x['DEPT']
    thresh = np.round(d.mean() + d.std() / 2)
    x['thresh'] = thresh
    return x[d<=thresh]

df_test = df.groupby(level=0).apply(get_test)
df_train = df.groupby(level=0).apply(get_train)
print('train', df_train.shape, 'test', df_test.shape)
print(f'Train {len(df_train)/len(df):.0%}, test {len(df_test)/len(df):.0%}')
# -

# We will be using depth and other measurements to determine the lithology. We dealt with the same problem in the tablular data. But in tabular data we only look at the measurements at each depth to find the class, while here we can look at the variations in the measurements as well.

# And add depth as a feature column:

# As usual we need to create a training and test set. here we will only use `15` wells for training and `15` for testing as using the entire dataset means we need to spend much longer time for training.



# We need to process the input and target data. The input data needs to be normalised with a standard scaler, and the output data needs to be converted from text to numbers. To convert text to numbers we use `LabelEncoder` from Scikit Learn.



# +
# # df['LITH_ID'] = encoder.transform(df.loc[:, "LITHOLOGY_GEOLINK"])
# # df['LITH_ID'] = encoder.transform(df.loc[:, "LITHOLOGY_GEOLINK"])
# df['LITH_ABV'] = df["LITHOLOGY_GEOLINK"].shift().fillna('Shaly Silt')
# df['LITH_ABV_INT'] = df['LITH_ABV'].values.codes
# df[['LITHOLOGY_GEOLINK', 'LITH_ABV']]
# -



# +
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()

# Make a encoder, that order by frequency
encoder = LabelEncoder()

# TODO need to embed prev val..., one hot, then append
encoder.classes_ = df["LITHOLOGY_GEOLINK"].values.categories # np.array(counts.index.values)
print(encoder.classes_)
feat_cols = ["CALI", "RHOB", "GR", "DTC", "RDEP", "RMED", "DEPT", "Well_int", "LITH_ABV_INT"]
scaler.fit(df[feat_cols].values)
# -

# `LabelEncoder` converts each type to a value.

encoder.transform(["Shaly Silt"])

# Now we can check the types at various depths:

# TODO nice plot, vertical, facies colors
plt.figure(figsize=(16, 5))
idx = 0
x = df.loc[wells[idx], "DEPT"]
y = encoder.transform(df.loc[wells[idx], "LITHOLOGY_GEOLINK"])
plt.plot(x, y)

# The output of a classification model is a value for each type. The type with the highest value is the one the model thinks is most likely to be associated with the input data. Therefore, the output size of the model should be the number of types.

output_size = len(df["LITHOLOGY_GEOLINK"].unique())

# Let's create training and test set, similar to what we had in multivariate time series. The only difference is here for each sequence of values we want the model to predict a value for each type.



# +
seq_length = 100

x_train = []
y_train = []
features = scaler.transform(df_train.loc[:, feat_cols].values)
targets = encoder.transform(df_train.loc[:, "LITHOLOGY_GEOLINK"])
for i in range(len(targets) - seq_length):
    xi = features[i : i + seq_length, :]
    yi = targets[i + seq_length - 1]
    x_train.append(xi)
    y_train.append(yi)

x_test = []
y_test = []
features = scaler.transform(df_test.loc[:, feat_cols].values)
targets = encoder.transform(df_test.loc[:, "LITHOLOGY_GEOLINK"])
for i in range(len(targets) - seq_length):
    xi = features[i : i + seq_length, :]
    yi = targets[i + seq_length - 1]
    x_test.append(xi)
    y_test.append(yi)


# +
# x_test.mean(), x_test.std()
# -

# It is important that we make sure the training and test set have close distribution. For instance, if there is a certain type in test data that doesn't exist in training data, the model will not be able to predict it.

# +
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    return x

def show_distribution(y):
    y = to_numpy(y)
    plt.hist(y, output_size * 2)
    plt.xticks(ticks=range(len(encoder.classes_)), labels=encoder.classes_, rotation=90)


# -

show_distribution(y_train)

show_distribution(y_test)



device = "cuda" if torch.cuda.is_available() else "cpu"
x_train = torch.Tensor(x_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
x_test = torch.Tensor(x_test).to(device)
y_test = torch.LongTensor(y_test).to(device)
x_train.shape, y_train.shape, x_test.shape, device

# We can still use the same class for the model. Here the input size is the number of features we are using to predict the type, and output size is the number of types. 

model = LSTM(
    input_size=len(feat_cols),
    hidden_size=16,
    num_layers=3,
    output_size=output_size,
)
model = model.to(device)
model



unique, counts = np.unique(to_numpy(y_train), return_counts=True)
weight = torch.from_numpy(1/(counts+1e3))
weight = None

optimizer = optim.Adam(model.parameters(), lr=0.001)
torch.nn.CrossEntropyLoss(weight=weight)
loss_func = torch.nn.CrossEntropyLoss()

from sklearn.metrics import accuracy_score, f1_score


def training_loop(epochs=1, bs=128):
    pbar1 = tqdm(total=epochs)
    pbar2 = tqdm(total=len(x_train) // bs)
    all_losses = []
    all_f1s = []
    try:
        for epoch in range(epochs):
            model.train()
            training_loss = []
            training_f1 = []
            pbar2.reset(len(x_train) // bs)
            for i in range(0, len(x_train), bs):
                optimizer.zero_grad()
                preds = model(x_train[i : i + bs, ...])
                loss = loss_func(preds, y_train[i : i + bs])
                loss.backward()
                optimizer.step()
                training_loss.append(loss.item())
                f1 = f1_score(
                    to_numpy(y_train[i : i + bs]), to_numpy(preds).argmax(-1), average='weighted'
                )
                training_f1.append(f1)
                pbar2.update(1)
#                 print(
#                     f"Epoch {epoch+1}/{epochs}: Loss = {loss.item():.2f}, f1 = {f1:.3f}",
#                     end="\r",
#                     flush=True,
#                 )

            print(
                f"Epoch {epoch+1}/{epochs}: Training Loss = {np.mean(training_loss):.2f}, f1 = {np.mean(f1):.3f}"
            )

            model.eval()
            test_loss = []
            test_f1 = []
            pbar2.reset(len(x_test) // bs)
            for i in range(0, len(x_test), bs):
                preds = model(x_test[i : i + bs, ...])
                loss = loss_func(preds, y_test[i : i + bs])
                test_loss.append(loss.item())
                f1 = f1_score(
                    to_numpy(y_test[i : i + bs]), to_numpy(preds.argmax(-1)), average='weighted'
                )
                test_f1.append(f1)
                pbar2.update(1)
            print(
                f"Epoch {epoch+1}/{epochs}: Test Loss = {np.mean(test_loss):.2f}, f1 = {np.mean(test_f1):.3f}"
            )
            print("-" * 50)
            all_losses.append([np.mean(training_loss), np.mean(test_loss)])
            all_f1s.append([np.mean(training_f1), np.mean(test_f1)])
            pbar1.update(1)
    except KeyboardInterrupt:
        pass

    # Visualising the results
    all_losses = np.array(all_losses)
    plt.plot(all_losses[:, 0], label="Training")
    plt.plot(all_losses[:, 1], label="Test")
    plt.title("Loss")
    plt.legend()
    plt.figure()
    all_f1s = np.array(all_f1s)
    plt.plot(all_f1s[:, 0], label="Training")
    plt.plot(all_f1s[:, 1], label="Test")
    plt.title("f1")
    plt.legend()



# Let's train for 10 epochs

training_loop(30, 1024)

preds = to_numpy(model(x_test).argmax(axis=-1))
true = to_numpy(y_test)

# Baseline
f1_score(true[1:],
    true[:-1:], average='weighted')

# ours
f1_score(true,
    preds, average='weighted')

plt.hist(preds, bins=output_size * 2)
plt.title("Model Predictions")
plt.xticks(np.arange(output_size))
plt.figure()
plt.hist(true, bins=output_size * 2)
plt.title("Targets")
plt.xticks(np.arange(output_size))
1

# +
import pandas as pd
import sklearn.metrics
import numpy as np

def classification_report(*args, **kwargs):
    out_df = pd.DataFrame(sklearn.metrics.classification_report(*args, **kwargs, output_dict=True)).T
    # Order cols
    out_df[["precision","recall","f1-score","support"]]  
    # Round
    out_df[["precision","recall","f1-score"]]= out_df[["precision","recall","f1-score"]].apply(lambda x: round(x,2))
    out_df[["support"]]= out_df[["support"]].apply(lambda x: x.astype(np.int))
    return out_df


# -

df_report = classification_report(preds, true, labels=range(len(encoder.classes_)), target_names=encoder.classes_)
df_report

# +
cm = sklearn.metrics.confusion_matrix(preds, true, labels=range(len(encoder.classes_)))

plt.figure(figsize=(20, 20))
plt.title('Confusion Matrix')
ax=plt.gca()
disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=encoder.classes_)
disp.plot(ax=ax, xticks_rotation=90)
plt.show()
# -



# As we saw in the previous examples the model requires training over 100s of epochs to reach the best results. However, in this example due to large size of dataset and the model we stopped after `10` epochs. Try increasing the number of epochs to see how it will affect the accuracy.<br>
#
# Obviously the model right now is not performing well. But there are number ways we can improve it:
# 1. Training for longer. instead of stopping after `10` epochs go for longer.
# 2. Increase the hidden state size.
# 3. Increase the size of training data by adding data from more wells to training. 
# 4. Increase the size of the sequences so the model get to look further in the history.
#
# #### Exercise 2
# Try one of the options above to improve the model.

# +
# Code Here
# -

# Let's have a look at model's predictions.



# The distribution of data shows that the model is too focused on type `17` and `18` which is why we are having low accuracy.

#
# <div class="alert alert-success">
#     
# ### Solutions  
# <details><summary>See solutions</summary>
#
# <details><summary>Exercise 1</summary>
# <b>Increase sequence length to 18</b>
#
# ```Python
# seq_length = 18
# x, y = create_seq_data(data,seq_length)
# xtrain = x[:100,:,:]
# ytrain = y[:100,:]
# xtest = x[100:,:,:]
# ytest = y[100:,:]
# model = LSTM(1, 50, 1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# train_model(model, xtrain, ytrain, 500)
# ```
# <b>Change model size to 100</b>
# ```Python
# model = LSTM(1, 100, 1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# train_model(model, xtrain, ytrain, 500)
# ```
#
# </details>
# <details><summary>Exercise 2</summary>
#     <b>Larger hidden size</b>
#     
# ```Python
# model = LSTM(
#     input_size=len(feat_cols),
#     hidden_size=400,
#     num_layers=1,
#     output_size=output_size,
# ).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# training_loop(10, 1024)
# ```
#
# <b>Train longer</b>
#     
# ```Python
# model = LSTM(
#     input_size=len(feat_cols),
#     hidden_size=200,
#     num_layers=1,
#     output_size=output_size,
# ).to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# training_loop(20, 1024)
# ```
#  
# </details>
# </details>
# </div>

# ## Further Reading
# - [Introduction to RNN](http://slazebni.cs.illinois.edu/spring17/lec02_rnn.pdf)
# - [A friendly introduction to Recurrent Neural Networks](https://www.youtube.com/watch?v=UNmqTiOnRfg)
# - [Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM)](https://www.youtube.com/watch?v=WCUNPb-5EYI&t=97s)
# - [Introduction to LSTM](https://medium.com/x8-the-ai-community/a-7-minute-introduction-to-lstm-5e1480e6f52a)
# - [LSTM and GRU](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
# - [Time Series Prediction with LSTM](https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/)
# - [Building RNN from scratch](https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79)
#
