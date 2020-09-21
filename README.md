# deep_ml_curriculum

Data science and Machine Learning training for Oil and Gas.

Teaches using some oil and gas specific datasets and examples such as well logs, seismic interpretations, geospatial plotting, productions plotting, and more.


<div>
<img src="reports/figures/LSTM_facies_pred.png" alt="Facies prediction with LSTM" width="300"/><img src="reports/figures/unsupervised.png" alt="drawing" width="300"/><img src="reports/figures/TSF.png" alt="Time series forecasting" width="300"/>
</div>

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Course notebooks. a is reserved for a python course. b is data science, c is machine learning
    │   ├── b01_SQL        <- Start of DS
    │   ├── b02_Advanced_Pandas
    │   ├── b03_Data_Visualisation
    │   ├── b04_Geopandas
    │   ├── b05_Interactive_Plotting
    │   ├── b06_Time_Series_Analysis
    │   ├── b07_Time_Series_Forcasting
    │   ├── b08_DS_Basics
    │   ├── b09_Supervised_Learning_Part_1
    │   ├── b10_Evaluation_Metrics
    │   ├── b11_Unsupervised
    │   ├── b12_Final_Project
    │   ├── c00_Introduction_to_Machine_Learning.pptx
    │   ├── c01_Intro_to_NN_Part_1
    │   ├── c02_Intro_to_NN_Part_2
    │   ├── c03_Finetuning
    │   ├── c04_Tabular_Data
    │   ├── c05_Big_Data
    │   ├── c06_Recurrent_Neural_Networks
    │   ├── c07_Hyperparameter_Optimization
    │   ├── c08_Autoencoders
    │   ├── c09_Object_Detection
    │   ├── c10_GANs
    │   └── z00_Data_prep  <- Preperation of datasets
    │
    ├── requirements       <- The requirements files for reproducing the analysis environment, e.g.
    │                         generated with `make doc_reqs`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── deep_ml_curriculum <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py




# Setup the data

The data is stored on a public (read only) s3 bucket

```
git clone git@github.com:3springs/deep_ml_curriculum.git
cd <project root>
# install the module, as an editable pip module
pip install -e .
# pull raw the data from public s3 bucket
# aws s3 sync s3://deep-ml-curriculum-data/data/ data/
# pull processed (smaller) data from s3
aws s3 sync s3://deep-ml-curriculum-data/data/processed/ data/processed/
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->deep_ml_curriculum
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/elcronos"><img src="https://avatars1.githubusercontent.com/u/9023043?v=4" width="100px;" alt=""/><br /><sub><b>Camilo</b></sub></a><br /><a href="https://github.com/3springs/deep_ml_curriculum/commits?author=elcronos" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/pooyad359"><img src="https://avatars1.githubusercontent.com/u/5551405?v=4" width="100px;" alt=""/><br /><sub><b>Pooya</b></sub></a><br /><a href="https://github.com/3springs/deep_ml_curriculum/commits?author=pooyad359" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/wassname"><img src="https://avatars1.githubusercontent.com/u/1103714?v=4" width="100px;" alt=""/><br /><sub><b>Mike C</b></sub></a><br /><a href="https://github.com/3springs/deep_ml_curriculum/commits?author=wassname" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Assistedevolution"><img src="https://avatars1.githubusercontent.com/u/18102704?v=4" width="100px;" alt=""/><br /><sub><b>Sean Driver</b></sub></a><br /><a href="https://github.com/3springs/deep_ml_curriculum/commits?author=Assistedevolution" title="projectManagement">📆</a></td>
    <td align="center"><a href="https://github.com/the-winter"><img src="https://avatars1.githubusercontent.com/u/19483860?v=4" width="100px;" alt=""/><br /><sub><b>the-winter</b></sub></a><br /><a href="https://github.com/3springs/deep_ml_curriculum/commits?author=the-winter" title="review">📆</a></td>

  </tr>
</table>
<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

# Credits

Many of the datasets or notebooks are based on resources that were generously made open source by the authors. These are aknowledged either in a readme file associated with the data, in the notebook, or at the end of the notebook.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

