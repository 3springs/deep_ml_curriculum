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
#     display_name: 'Python 3.7.3 64-bit (''jup3.7.3'': venv)'
#     language: python
#     name: python37364bitjup373venv303c3eb08efe4501baa24424ed3eb0f3
# ---

# Data source: https://dataunderground.org/dataset/landmass-f3
# - Credits to researchers at Georgia Tech, Agile Geoscience
# - License CCbySA

import h5py
from tqdm.auto import tqdm
from pathlib import Path
import torch
# %pylab inline

# # View

data_in = Path("../../data/raw/landmass-f3/")
data_out = Path("../../data/processed/landmass-f3/")

with h5py.File(data_in / "Landmass_CC-BY-SA.h5", "r") as f:
    print(list(f.attrs))
    print(list(f))
    for k in ["LANDMASS-1", "LANDMASS-2"]:
        print(k, list(f[k]))

f = h5py.File(data_in / "Landmass_CC-BY-SA.h5", "r")
nrows = 2
ncols = 4
for key in f["LANDMASS-1"].keys():
    d = f["LANDMASS-1"][key][: nrows * ncols, :, :, 0]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 4))

    for i in range(ncols * nrows):

        ax = axs[i // ncols][i % ncols]
        ax.imshow(d[i])
        ax.set_xticks([])
        ax.set_yticks([])

    #     plt.tight_layout()
    plt.suptitle(f"{key}")
    plt.show()

f = h5py.File(data_in / "Landmass_CC-BY-SA.h5", "r")
nrows = 2
ncols = 4
for key in f["LANDMASS-2"].keys():
    d = f["LANDMASS-2"][key][: nrows * ncols, :, :, 0]

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3))

    for i in range(ncols * nrows):

        ax = axs[i // ncols][i % ncols]
        ax.imshow(d[i])
        ax.set_xticks([])
        ax.set_yticks([])

    #     plt.tight_layout()
    plt.suptitle(f"{key}")
    plt.show()

# # Split

from sklearn.model_selection import train_test_split
import torch



# +
f = h5py.File(data_in / "Landmass_CC-BY-SA.h5", "r")
classes = ["Chaotic Horizon", "Fault", "Horizon", "Salt Dome"]
X = []
Y = []
for i, key in enumerate(tqdm(classes)):
    x = np.array(f["LANDMASS-1"][key].value)

    # convert to uint8
    x = (x + 1) / 2
    x *= 255
    x = x.astype(np.uint8)

    y = np.ones(x.shape[0]) * i
    X.append(x)
    Y.append(y)
    print(i, key, x.shape, x.min(), x.max())

X = np.concatenate(X, 0).astype(np.uint8)
Y = np.concatenate(Y, 0).astype(np.uint8)
X.shape, Y.shape

# +
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)

with (data_out / "LandmassF3Patches/processed/training.pt").open("wb") as fo:
    torch.save((torch.from_numpy(X_train)[:, :, :, 0], torch.from_numpy(y_train)), fo)
    print(fo)

with (data_out / "LandmassF3Patches/processed/test.pt").open("wb") as fo:
    torch.save((torch.from_numpy(X_test)[:, :, :, 0], torch.from_numpy(y_test)), fo)
    print(fo)
# -

np.split(X_train, 26)



# +
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)
X_train = np.concatenate(
    [
        X_train[:, :26, :26, 0],
        X_train[:, 26 * 1 : 26 * 2, 26 * 1 : 26 * 2, 0],
        X_train[:, 26 * 2 : 26 * 3, 26 * 2 : 26 * 3, 0],
    ]
)
y_train = np.concatenate([y_train, y_train, y_train])
y_test = np.concatenate([y_test, y_test, y_test])
X_test = np.concatenate(
    [
        X_test[:, :26, :26, 0],
        X_test[:, 26 * 1 : 26 * 2, 26 * 1 : 26 * 2, 0],
        X_test[:, 26 * 2 : 26 * 3, 26 * 2 : 26 * 3, 0],
    ]
)
print(X_train.shape, y_train.shape)
with (data_out / "LandmassF3PatchesMini/processed/training.pt").open("wb") as fo:
    torch.save((torch.from_numpy(X_train), torch.from_numpy(y_train)), fo)
    print(fo)

with (data_out / "LandmassF3PatchesMini/processed/test.pt").open("wb") as fo:
    torch.save((torch.from_numpy(X_test), torch.from_numpy(y_test)), fo)
    print(fo)
# -

torch.load(data_out / "LandmassF3PatchesMini/processed/training.pt")[0].shape

# # Load

# +
from torchvision.datasets import MNIST

mnist = MNIST("/tmp/mnist", download=True)
x, y = mnist[5]
np.array(x).shape

# +
from deep_ml_curriculum.data.landmass_f3 import LandmassF3Patches, LandmassF3PatchesMini
from deep_ml_curriculum.config import project_dir


landmassf3 = LandmassF3Patches(project_dir / "data/processed/landmass-f3")
print(landmassf3)
x, y = landmassf3[4]
print(landmassf3.classes[y])
print(np.array(x).shape)
x
# -

# WARNING This may be too small to see patterns, which may lower accuracy a lot on an already hard task
landmassf3mini = LandmassF3PatchesMini(project_dir / "data/processed/landmass-f3")
print(landmassf3mini)
x, y = landmassf3mini[40]
print(landmassf3.classes[y])
print(np.array(x).shape)
x






