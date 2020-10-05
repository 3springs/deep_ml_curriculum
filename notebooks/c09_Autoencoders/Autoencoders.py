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

# # Autoencoders
# ## What are autoencoders?
# Autoencoders are networks which have the same input and output. A set of data is fed to these networks and they are expected to recreate the input. However, what makes autoencoders interesting is that they compress the information into lower number of dimensions (a.k.a latent space) and then recreate the input using those dimensions. They can be used for dimensionality reduction similar to PCA, t-SNE, and Umap. Some of the advantages of using autoencoders compared to some of the other techniques are:
# - Flexibility: You can design the network based on what the problem demands.
# - Reversibility: Unlike methods such as t-SNE and UMAP you can convert data back to the initial space.
# - Non-linearity: Unlike linear methods such as PCA, it is capable of using non-linear transformation.
#
# ## Structure
# Autoencoders have two main components:
# 1. Encoder: Converts data to latent space.
# 2. Decoder: Converts the data back from latent space to its initial space.
#
# The architecture looks similar to the image below:
# <img src='./images/nn.svg' style='height:50rem'>
#
# We pass the input through the model and it will compress and decompress the input and returns a result. Then we compare the output of the model with the original input. To check how close the output is to the original input we use a loss function.

# Let's start by importing the required libraries.

# +
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms

from PIL import Image
from pathlib import Path
import numpy as np
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
# -

# ## Problem Description
# We are going to start with a simple problem. We will use MNIST dataset which is a collection of hand-written digits as 28x28 pixel images. We are going to use autoencoder to compress each image into only two values and then reconstruct the image. When the model is trained we will have a look at the reconstructed images as well as latent space values.

# First we need to create a `Dataset` class. The `Dataset` class reads the data from file and returns data points when we need them. The advantage of using a `Dataset` is that we can adjust it based on what we need for each problem. If we are not dealing with large amount of data we can decide to keep everything in RAM so it is ready use. But if we are dealing with a few gigabytes of data we might need to open the file only when we need them.<br>
# The MNIST data set is not large so we can easily fit it into memory. In the `Dataset` class we define a few methods:
# - `__init__`: What information is required to create the object and how this information is saved.
# - `__len__`: Returns the number of data points (images) when we use `len()` function.
# - `__getitem__`: We can define how indexing would work for this class.
#
# We are going to define a couple of custom functions for convinience:
# - `show`: to see the image.
# - `sample`: which returns a random sample of the data.

# +
path = Path("../../data/processed/MNIST/")


class DigitsDataset(Dataset):
    def __init__(self, path, transform=None):

        self.root_dir = Path(path)
        self.transform = transform
        data = pd.read_csv(path)
        if "label" in data.columns:
            self.x = data.drop(columns=["label"]).values
            self.y = data["label"].values
        else:
            self.x = data.values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        output = self.x[idx] / 255
        if self.transform:
            output = self.transform(output)
        return output

    def show(self, idx):
        plt.figure(figsize=(2, 2))
        plt.imshow(self.x[idx].reshape((28, 28)), "gray")

    def sample(self, n):
        idx = np.random.randint(0, len(self), n)
        return self[idx]


class ToTensor(object):
    def __call__(self, data):
        return torch.FloatTensor(data)


# -

# __Note:__ We also defined a class called `ToTensor`. This class takes an input and converts it to pytorch tensor. 

# Now that we have a `Dataset` class, we can create a training and test dataset.

ds_train = DigitsDataset(path / "train.csv", transform=ToTensor())
ds_test = DigitsDataset(path / "test.csv", transform=ToTensor())

# Next step is to create a data loaders. The training process takes place at multiple steps. At each step, we choose a few images and feed them to the model. Then we calculate the loss value based on the output. Using the loss value we update the values in the model. We do this over and over until when we think the model is trained. Each of these steps are called a mini-batch and the number of images passed in at each mini-batch is called batch size. Dataloader's job is to go to the dataset and grab a mini-batch of images for training. To create a Dataloader we use a pytorch dataloder object.

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    ds_train, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)


# __Note:__ Shuffle tells the data loader whether the data needs to be shuffled at the end of each epoch. We do it for training to keep the input random. But we don't need to do it for testing since we only use the test dataset for evaluation.

# Now we need to create the model. The architecture we are going to use here is made of two linear layers for the encoder and two linear layers for the decoder.

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 2)
        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        return self.decode(z)


# If we have access to GPU, let's make sure we are using it.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Now we create an instance of the model.

model = AE().to(device)

# We also need to choose an optimiser. The optimiser use the loss value and it's gradients with respect to model parameters and tells us how much each value must be adjusted to have a better model.

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# And the final component is the loss function. Here we are going to use Binary Cross Entropy function.

def loss_bce(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")

    return BCE


# Let's define two functions one for executing a single epoch of training and one for evaluating the mdel using test data.<br>
# Notice the following steps in the training loop:
# 1. We make sure the data is in the right device (cpu or gpu)
# 2. We make sure that any saved gradient (derivative) is zeroed.
# 3. We pass a mini-batch of data into the model and grab the predictions.
# 4. We use the loss function to find out how close the model's output is to the actual image.
# 5. We use `loss.backward()` to claculate the derivative of loss with respect to model parameters.
# 6. We ask the optimiser to update model's parameters.

# +
def train(epoch, loss_function, log_interval=50):
    model.train()
    train_loss = 0
    for batch_idx, data in tqdm(enumerate(train_loader), leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = loss_function(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            pct = 100.0 * batch_idx / len(train_loader)
            l = loss.item() / len(data)
            print(
                '#{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  '.format(epoch, batch_idx * len(data), len(train_loader.dataset), pct, l),
                end="\r",
                flush=True,
            )
    print('#{} Train loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch, loss_function, log_interval=50):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), leave=False):
            data = data.to(device)
            recon_batch = model(data)
            test_loss += loss_function(recon_batch, data).item()

    test_loss /= len(test_loader.dataset)
    print('#{} Test loss: {:.4f}'.format(epoch, test_loss))


# -

# Now that all the components are ready, let's train the model for $10$ epochs.

epochs = 10
for epoch in range(1, epochs + 1):
    train(epoch, loss_bce)
    test(epoch, loss_bce)


# ## Results
# Now let's check out the model.

def cvt2image(tensor):
    return tensor.detach().cpu().numpy().reshape(28, 28)


# +
idx = np.random.randint(0, len(ds_test))

model.eval()
original = ds_train[[idx]]
result = model(original.to(device))
img = cvt2image(result[0])
plt.figure(figsize=(2, 2))
plt.imshow(img, "gray")
plt.title("Predicted")
ds_train.show(idx)
plt.title("Actual")
# -

# <font color='blue' size='4rem'>Run the cell above a few times and compare the predicted and actual images.</font>

# There are certainly some similarities but the predicted (reconstructed) images are not always very clear. We will shortly discuss how we can improve the model. But before that, let's have look at the latent space. The model is converting every image which has 784 values (28x28 pixels) to only 2 values. We can plot these two values for a few numbers.

res = model.encode(ds_train[:1000].to(device))
res = res.detach().cpu().numpy()

for i in range(10):
    idx = ds_train.y[:1000] == i
    plt.scatter(res[idx, 0], res[idx, 1], label=i)
plt.legend()


# Each color represents a number. Despite most numbers overlapping, we can still see some distictions, for instance between $1$ and other numbers. 

# ## Improving the model
# Obviously the model that we trained needs improvement as it is no recreating the images well enough. There are a few ways we can improve the model. One way is to create a deeper encoder and decoder. In the example above we ued only two layers for encoder and layers for decoder. This doesn't allow the model to comprehend complex relationships, especially in this scenario since we are working with images. By adding more layers we can give the model the opportunity to better differentiate between digits.
# Another way of making the model is using more dimensions in latent space. For instance, if instead of compressing each image into two values we could use ten values. This will allow the model to extract more features from the input which will make reconstructing the image easier. However, it must be noted that whole point of using autoencoder is to force the model to compress the information into as few dimensions as possible.

# ## Application
# Autoencoders are not only useful for dimensionality reduction. They are often used for other purposes as well, including:
# 1. __Denoising:__ We could add noise to the input and then feed it to the model and then compare the output with the original image (without noise). This approach will create a model which is capable of removing noise from the input.
# 2. __Anomaly Detection:__ When we train a model on specific set of data, the model learns how to recreate the dataset. As a result when there are uncommon instances in the data the model will not be able to recrate them very well. This behaviour is sometimes used as a technique to find anomalous data points. 

# # Variational Autoencoders
# Variational Autoencoders (VAE) are one of the variations of autoencoders. Unlike normal autoencoders which compress the data into a few values, VAEs tries to find the distribution of the data in latent space. As a result, the final model not only has the ability to recreate the input, but can also generate new outputs by sampling from the latent space distribution.

# Since VAE is a variation of autoencoder, it has a similar architecture. The main difference between the two is an additional layer between encoder and decoder which samples from latent space distribution.
# In a VAE, the encoder generates two values for each parameter in latent space. One represent the mean and one represents the standard deviation of the parameter. Then sampling layer uses these two numbers and generates random values from the same distribution. These values then are fed to decoder which will create an output similar to the input.

# Let's create a VAE model. We will use layers with the same size as the previous model. Notice for the second layer we have two linear layers, one to generate the mean and one to generate the log of variance which will be converted into standard deviation.

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 2)
        self.fc22 = nn.Linear(400, 2)
        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# The loss function is similar to what we used before, except we have an extra part. the extra equation is Kullback–Leibler divergence which measures difference between probability distributions.

def loss_bce_kld(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# We also need to slightly adjust the training loop since the loss function now takes four inputs.

# +
def train(epoch, loss_function, log_interval=50):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            pct = 100.0 * batch_idx / len(train_loader)
            l = loss.item() / len(data)
            print(
                '#{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  '.format(epoch, batch_idx * len(data), len(train_loader.dataset), pct, l),
                end="\r",
                flush=True,
            )
    print('#{} Train loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch, loss_function, log_interval=50):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('#{} Test loss: {:.4f}'.format(epoch, test_loss))


# -

epochs = 10
for epoch in tqdm(range(1, epochs + 1)):
    train(epoch, loss_bce_kld)
    test(epoch, loss_bce_kld)

# ## Saving and Loading Model

import pickle

with open("VAE.pk", "wb") as fp:
    pickle.dump(model.state_dict(), fp)

model.load_state_dict

model = VAE()
with open("VAE.pk", "rb") as fp:
    model.load_state_dict(pickle.load(fp))


# ## Results

def cvt2image(tensor):
    return tensor.detach().numpy().reshape(28, 28)


# +
idx = np.random.randint(0, len(ds_test))

model.eval()
original = ds_train[[idx]]
result = model(original)
img = cvt2image(result[0])
plt.figure(figsize=(2, 2))
plt.imshow(img, "gray")
plt.title("Predicted")
ds_train.show(idx)
plt.title("Actual")
# -

# Now let's plot the predicted mean of both parameters.

mu, logvar = model.encode(ds_train[:1000])

mu = mu.detach().numpy()

for i in range(10):
    idx = ds_train.y[:1000] == i
    plt.scatter(mu[idx, 0], mu[idx, 1], label=i)
plt.legend()

# If we compare this plot with the similar plot for normal autoencoder, we can see that VAE did a better job at creating clusters. The points for each digits are closer together compared to previous model. However, there is still room for improvement. 

# ## Exercise 1
# Create a new VAE but this time use a deeper network. Note, everything else (loss function, dataloaders, training loops, etc.) will stay the same only the model will change. The example above was using these sizes: 784 --> 400 --> 2 --> 400 --> 784
# <br>Try a new model which uses these size: 784 --> 400 --> 80 --> 2 --> 80 --> 400 --> 784 

# +
# Create the model definition

# +
# Insert Training loop here

# +
# Visualise the results
# -

# # Convolutional Autoencoder
# So far we only used __Linear layers__ (also called Dense layers) in our network. The problem with using linear layers is that don't understand the spatial relationship between pixels. For a linear layer all the inputs are the same. But since we know that there is a closer relationship between the pixels located near each other, using a convolutional layer is more appropriate. 

# This time we are going to apply a Convolutional Variational Autoencoder on 2D DeepRock dataset.

path = Path(
    "../../data/processed/deep-rock-sr/DeepRockSR-2D/coal2D/coal2D_train_LR_default_X4/"
)

img_list = os.listdir(path)

# Let's have look at a random image from this dataset.

rand_file = np.random.choice(img_list)
img = Image.open(path / rand_file)

img

np.array(img).shape


# Let's create our model. Same as before the model has three main sections:
# 1. __Encoder:__ After each layer in the encoder we decrease the size of output and increase the number of channels. 
# 2. __Decoder:__ It acts as opposit of encoder. At each layer we increase the size of output and decrease the number of channels.
# 3. __Reparametrisation (Sampler):__ Gets a mean and log of variance of the distribution and generates a new datapoint.

class CVAE(nn.Module):
    def __init__(self, n_latent=32):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

        self.fc11 = nn.Linear(1024, n_latent)
        self.fc12 = nn.Linear(1024, n_latent)
        self.fc2 = nn.Linear(n_latent, 1024)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn_like(mu)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc11(h), self.fc12(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc2(z)
        z = z.view(z.size(0), 1024, 1, 1)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


# We need to create a new dataset and dataloader object for this dataset. Note that in this case we are not loading all the data in the memory, but at each mini-batch we only open the images that we need.

# +
class RocksDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.transform = transform
        self.image_paths = paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx]
        image = Image.open(img_name).resize((64, 64))
        image = np.array(image) / 255
        if self.transform:
            image = self.transform(image)

        return image


class ToTensor(object):
    def __call__(self, image):
        return torch.FloatTensor(image).permute((2, 0, 1))


# -

# Let's split the data into training and test set.

image_list = np.array([i for i in path.iterdir()])
is_train = np.random.binomial(1, 0.8, len(image_list))
list_train = image_list[is_train == 1]
list_test = image_list[is_train == 0]

# Create dataset and dataloader for training and test sets.

bs = 32
ds_train = RocksDataset(list_train, transform=ToTensor())
train_loader = torch.utils.data.DataLoader(ds_train, bs, shuffle=True)
ds_test = RocksDataset(list_test, transform=ToTensor())
test_loader = torch.utils.data.DataLoader(ds_test, bs)


def loss_bce_kld(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Create an instance of the model and choose the optimiser.

model = CVAE()
model.to(device)
optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())


# Create a training and test loop.

# +
def train(epoch, loss_function, log_interval=20):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            pct = 100.0 * batch_idx / len(train_loader)
            l = loss.item() / len(data)
            print(
                '#{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  '.format(epoch, batch_idx * len(data), len(train_loader.dataset), pct, l),
                end="\r",
                flush=True,
            )
    print('#{} Train loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch, loss_function, log_interval=20):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)
    print('#{} Test loss: {:.4f}'.format(epoch, test_loss))


# -

# ## Train the model
# And finally train the model. Since we are training a larger network from scratch we need to let the model train for longer.

from time import perf_counter

# <div class='alert alert-warning'>Training can take a long time. Each epoch takes 4 seconds on RTX 2070. Depending on your GPU it might take shorter or longer. Feel free to adjust the number of epochs accordingly.</div>

epochs = 30
for epoch in tqdm(range(1, epochs + 1)):
    t = perf_counter()
    train(epoch, loss_bce_kld)
    test(epoch, loss_bce_kld)
    print('Epoch#{} took {:.1f} seconds'.format(epoch, perf_counter() - t))


# ## Results
# When the model is trained we can have a look at the results.

def show_result(sample):
    sample = sample.unsqueeze(0).to(device)
    model.to(device)
    model.eval()
    output, _, _ = model(sample)  # we only grab the reconstructed image
    loss = F.binary_cross_entropy(output, sample, reduction="sum")
    print('Loss: {:.2f}'.format(loss.item()))

    # return the data to cpu for visualisation
    output = output.squeeze(0).permute(1, 2, 0).to("cpu")
    output = output.detach().numpy()
    sample = sample.squeeze(0).permute(1, 2, 0).to("cpu")
    sample = sample.numpy()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(sample)
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title("Reconstructed")


# grab an image from test set using the index
idx = 0
sample = ds_test[idx]
show_result(sample)

# We can play with learning rate, batch size, number of variabes in latent space and see if it improves the model.
# You can also change the size of the input images. However, since the model is designed for the images of size 64 by 64, if you want to use it for other sizes of image, you need to adjust the model as well.
#

# # Semantic Segmentation with Unet
# Semantic segmentation is very similar to image classification. In image classification we give an image to the model and we expect the model to tell us what category the image belongs to. In Semantic segmentation we ask the model to classify the image, but on pixel level. We expect the model to classify every pixel of the image and tell us whether it belongs to one of the categories or not.<br>
# Let's say we have the image below and we want to segment humans in the image:<br>
# <img src='./images/tennis.png' width='50%'>
#
# A segmentation model looks at this image and returns a binary image (also called mask) which tells us which pixels in the image above belong to a human.<br>
# <img src='./images/mask.png' width='50%'>
#
# One of the most common architectures for image segmentation is called __U-net__. U-net is very similar to an autoencoder. It has an encoder and a decoder. But what makes U-net so powerful is its skip-connections. In a U-net the middle layers of the encoder is connected to he middle layers of the decoder. This allows the model to remember the details of the original image (using skip connections) and at the same time get a good understanding of the content of the image (using encoder and decoder).

# ## Pretrained model
# We are not going to write a U-net model from scratch like we did with autoencoders. Instead we use the pretrained models from `torch.hub`. This way, we can use the models that are already trained for this task (even though they might be trained on different datasets). If the model is trained on a similar data we can use it out of the box, if not we can retrain it using out own datasets. This methods is usually more effective than training a model from scratch.

import torch.hub
import torchvision

# For this section we are using a subset of COCO dataset. This subset has various photos and the goal is to find people in the photos.

# We can simply look at [pytorch hub](https://pytorch.org/hub/) and download the model we need. The code below downloads [DEEPLABV3-RESNET101](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) which suits the data we have.

try:
    model = torch.hub.load('pytorch/vision:v0.7.0', 'deeplabv3_resnet101', pretrained=True)
except Exception as e:
    # The model is also saved in the following path (torch==1.4.0) and we can load it directly.
    model = torch.load("../../data/processed/models/Deeplabv3.pth")



# ### Inference
# We can test the model on our data. While our data is selected for its human subjects, the model is trained on 20 different categories. Therefore, the output of the model may include subjects other than humans as well.

imgpath = "../../data/processed/COCO_sample/samplecoco2014"
maskpath = "../../data/processed/COCO_sample/masks2014/"
filename = np.random.choice(os.listdir(imgpath))
maskname = filename.replace("jpg", "png")
input_image = Image.open(os.path.join(imgpath, filename))
mask = Image.open(os.path.join(maskpath, maskname))

# +
from torchvision import transforms

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
input_batch = input_batch.to(device)
model.to(device)

with torch.no_grad():
    output = model(input_batch)["out"][0]
output_predictions = output.argmax(0).to("cpu")
# -

input_image

# +
plt.figure(figsize=(5, 15))

plt.subplot(3, 1, 1)
plt.title("Input Image")
plt.imshow(input_image)

plt.subplot(3, 1, 2)
plt.title("Prediction")
plt.imshow(output_predictions.numpy())

plt.subplot(3, 1, 3)
plt.title("Ground Truth")
plt.imshow(mask)


# -

# ## Train a model
# But what if the target in our dataset was different from the pretrained model? In that case the model won't be able to make a good prediction as it doesn't know our target object. But this doesn't mean that the pretrained model will be entirely useless. We can still use it, but it needs to be refined. Also the pretrained model is designed for 20 categories. But we might have smaller or larger number of categories.<br>
# Either way we need to train the model on our dataset. And the steps are similar to before:
# 1. Create a dataset
# 2. Create a data loader
# 3. Create the model
# 4. Select the optimiser
# 5. train the model

# ### Create a dataset

# +
class HumanDataset(Dataset):
    def __init__(self, path, subdirs, filenames, transform=None):
        self.transform = transform
        self.image_paths = [os.path.join(path, subdirs[0], f) for f in filenames]
        self.mask_paths = [
            os.path.join(path, subdirs[1], f.replace("jpg", "png")) for f in filenames
        ]
        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.image_paths[idx]).resize((128, 128))
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.array(image) / 255

        mask = Image.open(self.mask_paths[idx]).resize((128, 128))
        mask = np.array(mask) / 255
        if self.transform:

            image = self.transform(image).permute((2, 0, 1))
            image = self.normalizer(image)
            mask = self.transform(mask).unsqueeze(0)
        return image, mask


class ToTensor(object):
    def __call__(self, image):
        return torch.FloatTensor(image)


# -

# ### Create a DataLoader

# +
tsfm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

filenames = os.listdir("../../data/processed/COCO_sample/samplecoco2014/")
ds = HumanDataset(
    path="../../data/processed/COCO_sample/",
    subdirs=["samplecoco2014", "masks2014"],
    filenames=filenames,
    transform=ToTensor(),
)
dl = DataLoader(ds, batch_size=16, shuffle=True)
# -

# ### Create the model

# +
## TODO mike put these offline
# torchvision.models.segmentation.segmentation.model_urls = {'fcn_resnet50_coco': None,
#  'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth',
#  'deeplabv3_resnet50_coco': None,
#  'deeplabv3_resnet101_coco': 'https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth

# +
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def DeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
    model.classifier = nn.Sequential(DeepLabHead(2048, outputchannels), nn.Sigmoid())
    return model


# -

# <div class='alert alert-danger'><b>When creating an instance of model access to internet is required for downloading the pretrained model. If you don't have access to internet you won't be able to run the rest of the code.</b></div>

model = DeepLabv3()
model.to(device)

# ### Select the Optimiser

optimizer = optim.Adam(model.parameters(), lr=0.001)

dl.dataset[40][0].shape

# ### Train the Model

# +
epochs = 5

model.to(device)
model.train()
train_loss = []
for epoch in range(epochs):
    for data, target in tqdm(dl):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = F.sigmoid(model(data)["out"])
        loss = F.binary_cross_entropy(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        print('loss: {:.2f}   '.format(loss.item()), end="\r", flush=True)
    print('Training Loss epoch#{}: {:.2f}'.format(epoch + 1, np.mean(train_loss)))
# -



# When the model is trained you can test it on the data similar to the [example](#Inference) we saw before.

# # Solution to Exercises

# ## Exercise 1
# <details><summary>Solution</summary>
#
# ```Python
# class VAE2(nn.Module):
#     def __init__(self):
#         super(VAE2, self).__init__()
#
#         self.fc1 = nn.Linear(784, 400)
#         self.fc2 = nn.Linear(400, 80)
#         self.fc31 = nn.Linear(80, 2)
#         self.fc32 = nn.Linear(80, 2)
#         self.fc4 = nn.Linear(2, 80)
#         self.fc5 = nn.Linear(80, 400)
#         self.fc6 = nn.Linear(400, 784)
#
#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         h2 = F.relu(self.fc2(h1))
#         return self.fc31(h2), self.fc32(h2)
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std
#
#     def decode(self, z):
#         h3 = F.relu(self.fc4(z))
#         h4 = F.relu(self.fc5(h3))
#         return torch.sigmoid(self.fc6(h4))
#
#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 784))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar
#
#     
# model = VAE2().to(device)
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# epochs = 10
# for epoch in range(1, epochs + 1):
#     train(epoch,loss_bce_kld)
#     test(epoch,loss_bce_kld)
#
# # visualisations
#
# model.eval()
# mu , logvar = model.encode(ds_train[:1000])
# mu = mu.detach().numpy()
# for i in range(10):
#     idx = ds_train.y[:1000]==i
#     plt.scatter(mu[idx,0],mu[idx,1],label = i)
# plt.legend()
#
# ```
#
# </details>

# # References
# - [Pytorch examples for VAE](https://github.com/pytorch/examples/tree/master/vae)

# # Further Reading
# - [Autoencoders Explained](https://www.youtube.com/watch?v=7mRfwaGGAPg)
# - [Introduction to Variational Autoencoders](https://www.youtube.com/watch?v=9zKuYvjFFS8&t=527s)
# - [U-net](https://www.youtube.com/watch?v=81AvQQnpG4Q)
# - [Understanding Semantic Segmentation](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)
# - [Understanding Variation Autoencoders](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
# - [Visualizing MNIST using a variational autoencoder](https://www.kaggle.com/rvislaywade/visualizing-mnist-using-a-variational-autoencoder)
