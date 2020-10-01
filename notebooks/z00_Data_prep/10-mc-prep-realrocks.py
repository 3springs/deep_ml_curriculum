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

# # Real rocks data set
#
#
# The data shown here is a small subset of 90000 m (90km) of depth indexed core image sticks from the Norwegian Continental Shelf. GeoProvider sells the entire dataset to interested parties as a petrel ready product and as stand alone database. 
#
# source: https://drive.google.com/drive/u/0/folders/1dUTsx1AvqGzqMTv9FFjgJspv5V-9DfOU
#
# ## Data Credits
#
# These depth referenced core images have been made available  to all under the Creative Commons Attribution 4.0 https://creativecommons.org/licenses/by/4.0/legalcode
#
# This data (RealRock) has been made available by a generous sponsoring of GeoProvider http://geoprovider.no/. 
# You must acknowledge Geoprovider with full website link in every publication or project using this data. For example 
# `Data are © GeoProvider https://geoprovider.no/ and licensed CC-BY 4.0`
#
# The data shown here is a small subset of 90000 m (90km) of depth indexed core image sticks from the Norwegian Continental Shelf. GeoProvider sells the entire dataset to interested parties as a petrel ready product and as stand alone database. Discounts are available for Universities. In addition all public available composite logs have been depth indexed and are available as a petrel ready product.
#
# The data assembly and most of the quality control has been carried out on the freelancing platform freelancer.org and had numerous key contributors. The wages paid in this project and timeline given to the freelancers were always honours and way above minimum standards in the respective countries of the contributors.
#
# Key Contributors were
#
# - Data Nation @Fingerprint (India)
#     - Excel working file combination, crossing t and dotting I. Cross checking. Creation of core image database
#     - Creation of composite log image database
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

datadir_in = Path("../../data/raw/real-rock-geoprovider/")
datadir_out = Path("../../data/processed/real-rock-geoprovider/")

# +
# load excel sheet
df_csv = pd.read_csv(
    datadir_in / "RealPore Por Perm Lithology data 1240 Wells Norway public.csv.zip",
    compression="zip",
)

df_csv = df_csv.drop(
    columns=["Data source file name", "seq numb", "NPDID", "Plug or sample number"]
)

df_csv = df_csv.dropna(
    how="any",
    subset=[
        "Well Name",
        "main lithology",
        "grain size",
        "gain density gr/cm3",
        "porosity best of available",
        "sorting",
    ],
)

num_cols = ["Measured Depth", "gain density gr/cm3", "porosity best of available"]
for col in num_cols:
    df_csv[col] = pd.to_numeric(df_csv[col], errors="coerce")

df_csv = df_csv.dropna(how="any", subset=num_cols)
# drop rows with junk

# df_csv = df_csv.dropna(how='any', subset=['NPDID', 'Well Name'])
df_csv["well_name"] = df_csv["Well Name"].str.replace("-", "_")
df_csv["md"] = df_csv["Measured Depth"].astype(int)
df_csv = df_csv.dropna(axis=1, thresh=0.9 * len(df_csv))
df_csv
# -

df_csv.info()

df_depths = df_csv.groupby(["well_name", "md"]).first()
df_depths

paths_images = sorted(
    (datadir_in / 'public_core_images_crop_3570m_mid_norway'.format()).glob("**/*.jpg")
)
len(paths_images)

# +
found = 0
norows = 0
nowell = 0
well_names = set(df_csv["well_name"])

(datadir_out / "images").mkdir(parents=True, exist_ok=True)
labels = []
for j, image_path in enumerate(tqdm(paths_images)):

    fs = image_path.stem.split("_")
    well_name = "_".join(fs[:3])
    depth_a = float(fs[-2].replace(",", "."))
    depth_b = float(fs[-1].replace(",", "."))

    if well_name in well_names:
        df_well = df_csv[df_csv.well_name == well_name].set_index("Measured Depth")
        df_well = df_well.sort_index()
        rows = df_well[depth_a:depth_b].copy()
        if len(rows) > 0:
            found += 1
            if found < 5:
                print(well_name, depth_a, depth_b)
                display(rows)
                display(PIL.Image.open(image_path))

            # take middle row
            label = rows.iloc[len(rows) // 2].copy()

            # record label
            label["image"] = image_path.name
            labels.append(label)

            # move image
            shutil.copy(image_path, datadir_out / "images" / image_path.name)
        else:
            norows += 1
    else:
        nowell += 0

len(labels)
# -

found, norows, nowell

# +
df_labels = pd.DataFrame(labels)
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df_labels, random_state=42)
df_train.to_parquet(datadir_out / "train.parquet", compression="gzip")
df_test.to_parquet(datadir_out / "df_test.parquet", compression="gzip")
len(df_train), len(df_test)
# -







# +
# def get_header_len(path_log):
#     """Find head length in las file."""
#     for i, line in enumerate(path_log.open()):
#         if line.startswith('~A'):
#             return i


# def read_realrocks_data(path_log: Path):
#     """Read las file from realrocks, with qoutes fields that have spaced in"""

#     # data is invalid due to spaces, read it seperatly
#     l = lasio.read(path_log,  ignore_data=True)

#     # read data
#     skiprows = get_header_len(path_log) + 1
#     null = l.well['NULL'].value
#     df_data = pd.read_table(path_log, skiprows=skiprows, header=None, quoting=0, sep='\s+', na_values=[null])
#     l.set_data(df_data.values, truncate=False)

#     return l

# +
# for j, dir_log in enumerate(sorted((datadir_in/'Finalized').glob(f'*/'))):
#     name1 = dir_log.stem.split('_', 2)[-1]
#     if name1 in well_names:
#         name = name1
#         path_log = dir_log / f'{name}_Conditioned.las'
#         path_logim = datadir_in/f'Finalized/ENCL_1_{name}/WB_MULTIPLE_WELLS__ROCK_AND_CORE__CORE_DESC_REPORT_4_ENCL_1.TIF'
#         paths_images = sorted((datadir_in / f'public_core_images_crop_3570m_mid_norway/{name}').glob('*.jpg'))

#         # Load las
#         l = read_realrocks_data(path_log)
#         display(l.curves)
#         df_l = l.df()

#         for i, f in enumerate(paths_images):
#             if i>4:
#                 break
#             depth_a, depth_b = f.stem.split('_')[-2:]
#             depth_a, depth_b = int(float(depth_a.replace(',', '.'))), int(float(depth_b.replace(',', '.')))

#             df_well = df_csv[df_csv['well_name']==name]
#             df_well = df_well.set_index('Measured Depth')
#             rows = df_well[depth_a:depth_b]
#             if len(rows):
#                 print(name, depth_a, depth_b)
#                 print(rows)
#                 display(df_l[depth_a:depth_b].iloc[0])
#                 display(PIL.Image.open(f))


#         if j>3:
#             break
