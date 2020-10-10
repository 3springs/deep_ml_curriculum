import nbformat
import pytest
import os
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path


def run_notebook(notebook_path):
    # https://www.blog.pythonlibrary.org/2018/10/16/testing-jupyter-notebooks/
    nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
    dirname = os.path.dirname(notebook_path)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    proc = ExecutePreprocessor(timeout=60, kernel_name="deep_ml_curriculum")
    proc.allow_errors = True

    proc.preprocess(nb, {"metadata": {"path": "/"}})
    output_path = os.path.join(dirname, "{}_all_output.ipynb".format(nb_name))

    with open(output_path, mode="wt") as f:
        nbformat.write(nb, f)
    errors = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if output.output_type == "error":
                    errors.append(output)
    return nb, errors


notebooks = [
    "notebooks/b01_SQL/SQLITE_Basics.ipynb",
    "notebooks/b02_Advanced_Pandas/Pandas.ipynb",
    "notebooks/b03_Data_Visualisation/DataVisualisation.ipynb",
    # "notebooks/b04_DS_Basics/MachineLearning_(DS_Basics).ipynb",
    # "notebooks/b05_Supervised_Learning/supervised_part1.ipynb",
    # "notebooks/b06_Evaluation_Metrics/Supervised_Part_2_(Evaluation_Metrics).ipynb",
    # "notebooks/b07_Selfsupervised/Unsupervised.ipynb",
    "notebooks/b08_Interactive_Plotting/01_Holoviews.ipynb",
    # "notebooks/b09_Time_Series_Analysis/TSA.ipynb",
    # "notebooks/b10_Time_Series_Forecasting/TSF.ipynb",
    # "notebooks/b11_Geopandas/Geospatial_plotting.ipynb",
    # "notebooks/b12_Final_Project/FinalProject.ipynb",
    # "notebooks/c01_Intro_to_NN_Part_1/Intro_to_NN_Part_1.ipynb",
    # "notebooks/c02_Intro_to_NN_Part_2/Intro_to_NN_Part_2.ipynb",
    # "notebooks/c03_Finetuning/Finetuning.ipynb",
    # "notebooks/c04_Tabular_Data/Tabular_Data_and_Embeddings.ipynb",
    # "notebooks/c05_Big_Data/Working_with_Big_Data.ipynb",
    # "notebooks/c06_Hyperparameter_Optimization/HyperparamOptimization.ipynb",
    # "notebooks/c07_Recurrent_Neural_Networks/02_mike-seq2seq_timeseries-simple.ipynb",
    # "notebooks/c07_Recurrent_Neural_Networks/Recurrent_Neural_Networks.ipynb",
    # "notebooks/c08_Object_Detection/Object_Detection.ipynb",
    # "notebooks/c09_Autoencoders/Autoencoders.ipynb",
    # "notebooks/c10_GANs/GANs.ipynb",
    # "notebooks/c11_Final_Project/FinalProject.ipynb",
]


def cpu_available():
    import torch
    assert torch.cuda.is_available()

def major_imports():
    import fbprophet
    import xarray
    import holoviews
    import datashader
    import torch
    import dask
    import pandas
    import sklearn
    import pillow
    import matplotlib
    import numpy


@pytest.mark.parametrize("notebook", notebooks)
def test_notebooks(notebook):
    print(notebook)
    nb, errors = run_notebook(notebook)
    print('errors', errors)
    assert len(errors) == 0

