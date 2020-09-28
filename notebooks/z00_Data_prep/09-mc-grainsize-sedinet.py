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

# From https://github.com/MARDAScience/SediNet/blob/master/notebooks/SediNet_Continuous_SievedSand_sieveplus4Prcs.ipynb
#
# This dataset has only 400 images, but they are large ~(3000x2000). You can to predict the PX values and plot it similar to the notebook above

# %pylab inline
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import lasio
from pprint import pprint
import PIL
import shutil
from IPython.display import display

data_in = Path("../../data/raw/SediNet")

csv_file = pd.read_csv(data_in / "grain_size_sieved_sands" / "data_pescadero_sieve.csv")
csv_file

for i, row in csv_file.sample(5).iterrows():
    break
im = PIL.Image.open(data_in / row["files"])
print(im.size, im, row)
im.resize((256, 256))

# TODO:
#     - copy only ones we use to processed
#     - break into smaller images


