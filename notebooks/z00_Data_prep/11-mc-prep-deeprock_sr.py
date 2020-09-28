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

# # DeepRockSR
# source https://www.digitalrocksportal.org/projects/215
#
# ## Data Description 
#
# A Diverse Super Resolution Dataset of Digital Rocks (DeepRockSR): Sandstone, Carbonate, and Coal.
#
# This dataset contains an organised and processed collection of greyscale digital rock images for the purpose of image super resolution training. In total, there are 12,000 2D images and 3,000 3D volumes contained in this dataset.
#
# Sandstone:
#
# Bentheimer Sandstone 1 https://www.digitalrocksportal.org/projects/211
# Bentheimer Sandstone 2 https://www.digitalrocksportal.org/projects/135
# Berea Sandstone https://www.digitalrocksportal.org/projects/135
# Leopard Sandstone https://www.digitalrocksportal.org/projects/135
# Gildehauser Sandstone https://www.digitalrocksportal.org/projects/134
# Wilcox Tight Sandstone https://www.digitalrocksportal.org/projects/6
#
# Carbonate:
#
# Estaillades Carbonate https://www.digitalrocksportal.org/projects/58
# Savonnieres Carbonate https://www.digitalrocksportal.org/projects/72
# Massangis Carbonate https://www.digitalrocksportal.org/projects/73
#
# Coal:
#
# Sheared Coal https://www.digitalrocksportal.org/projects/21
# Naturally Fractured Coal https://www.digitalrocksportal.org/projects/20
#
#
# The dataset is organised in a similar fashion to the DIV2K dataset : (https://data.vision.ee.ethz.ch/cvl/DIV2K/). To the knowledge of the authors, this dataset is the first of its kind to be used for the purpose of benchmarking Super Resolution algorithms in digital rock images. This dataset has been used thus far to test several Super Resolution Convolutional Neural Networks (SRCNN). Results are described in the Related Publications tab on this page.
#
# Please cite both this dataset (use its DOI) as well as the related journal publication.
#
# ### Data overview:
#
# The dataset is divided into:
#
# • 2D and 3D folders
# • Sandstone, Carbonate, and Coal Datasets
# • Combined shuffled Datasets
# • High Resolution, 2x and 4x downsampled datasets using the Matlab imresize and imresize3 functions with default settings and randomised settings
# • 2D HR images measure 500x500, cropped samples from the centre of the original cylindrical images, saved as PNG files
# • 9600 2D images are available for training, 1200 for validation, and 1200 for testing.
# • 3D HR volumes measure 100x100x100, cropped from the centre of the original cylindrical images, saved as octave and scipy readable MAT files (each array name is 'temp')
# • 2400 3D volumes are available for training, 300 for validation, and 300 for testing.
#

# %pylab inline
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import lasio
from pprint import pprint
import PIL
import shutil
from IPython.display import display

datadir_in = Path("../../data/raw/deep-rock-sr/")
datadir_out = Path("../../data/processed/deep-rock-sr/")

import torchvision

sorted(datadir_in.glob("**/*train_*"))



# +
# data_train = torchvision.datasets.ImageFolder(
#     '../../data/raw/deep-rock-sr/DeepRockSR-2D/',
#     is_valid_file=lambda f:('train_LR_default_X4' in f) and not ('shuffle' in f)
# )

# for i in np.random.choice(range(len(data_train)), 10):
#     x, y = data_train[i]
#     print(dict(i=i, y=y, classname=data_train.classes[y]))
#     display(x)

# +
# # copy small dataset to processed folder
# -

for p in tqdm(sorted(datadir_in.glob("**/*LR_default_X4*"))):
    if "shuffle" not in p.name:
        print(p)
        shutil.copytree(p, str(p).replace("raw", "processed"))
        pass

# # Load small 2d

data_train = torchvision.datasets.ImageFolder(
    "../../data/processed/deep-rock-sr/DeepRockSR-2D/",
    is_valid_file=lambda f: ("train_LR_default_X4" in f) and not ("shuffle" in f),
)
print(data_train)
print(data_train.classes)
for i in np.random.choice(range(len(data_train)), 5):
    x, y = data_train[i]
    print(dict(i=i, y=y, classname=data_train.classes[y]))
    display(x)

# # Load small 3d

# +
import h5py

load_mat = lambda f: np.array(h5py.File(f)["temp"])
data_train = torchvision.datasets.ImageFolder(
    "../../data/processed/deep-rock-sr/DeepRockSR-3D/",
    loader=load_mat,
    is_valid_file=lambda f: ("train_LR_default_X4" in f) and not ("shuffle" in f),
)
print(data_train)
print(data_train.classes)
for i in np.random.choice(range(len(data_train)), 5):
    x, y = data_train[i]
    print(dict(i=i, y=y, classname=data_train.classes[y]))
    plt.subplot(131)
    plt.title("y-z")
    plt.imshow(x[15, :, :])
    plt.subplot(132)
    plt.title("x-z")
    plt.imshow(x[:, 15])
    plt.subplot(133)
    plt.title("x-y")
    plt.imshow(x[:, :, 15])
    plt.show()
# -




