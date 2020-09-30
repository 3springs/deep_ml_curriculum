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

# # Introduction
#
# This notebook provides examples of some machine learning methods used in supervised learning. Please refer to the table below to navigate through the notebook.
#
# ## Table of Content
# 0. [Supervised Learning](#supervised)
# 1. [Libraries](#libraries)
# 2. [Dataset](#dataset)
# 1. [KNN](#knn)
# 2. [SVM](#svm)
# 3. [Decision Trees](#decision-trees)
# 4. [Random Forest](#random-forest)

# # 0. Supervised Learning <a name="supervised"></a>
#
# Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs [1]. It infers a function from labelled training data consisting of a set of training examples [2].
#
# ![ML](ml2.png)
# ![ML](ml1.png)
#
# Sources:
# - [Wikipedia](https://en.wikipedia.org/wiki/Supervised_learning#:~:text=Supervised%20learning%20is%20the%20machine,a%20set%20of%20training%20examples.)
# -  1. Stuart J. Russell, Peter Norvig (2010) Artificial Intelligence: A Modern Approach, Third Edition, Prentice Hall ISBN 9780136042594.
# -  2. Mehryar Mohri, Afshin Rostamizadeh, Ameet Talwalkar (2012) Foundations of Machine Learning, The MIT Press ISBN 9780262018258.

# ## 1. Importing libraries <a name="libraries"></a>

# +
import sklearn
#Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
#Classifiers
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# Preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Data visualisation
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# %matplotlib inline
# Hide all warnings
import warnings
warnings.filterwarnings('ignore') # warnings.filterwarnings(action='once')
# -

# ## 2. About the Geolink Dataset <a name="dataset"></a>
#
# This is a well dataset of Well's Petroleum wells drilled on the Norwegian continental shelf. It was released by the Norweigen Government, cleaned by Geolink, and loaded by LukasMosser. The full dataset has 221 wells, over 150 lithologies, with around 500 MB of data.
#
#
# You don't need to understand the data fully for this course, but here's a brief overview
#
# - Well - The well name
# - DEPT - Depth below the ground in meters
# - LITHOLOGY_GEOLINK - This is the facies or lithology, which means rock type. This is a label, made by multiple human experts by looking at the context, the well, maps of the area, and often picture of rock samples extracted from the well.
# - [Well logs](https://en.wikipedia.org/wiki/Well_logging): These are specialised measurements by instruments lowered down the well hole
#     - CALI - [Caliper log](https://en.wikipedia.org/wiki/Caliper_log), this measures the size of the well bore
#     - GR - [Gamma Ray](https://en.wikipedia.org/wiki/Gamma_ray_logging): Measure passive amount of high energy electromagnetic radiation naturally emitted from the rock
#     - RHOB - [Bulk Density](https://en.wikipedia.org/wiki/Density_logging): Measured active amount high energy electromagnetic radiation. This has a transmitting source of gamma rays
#     - DTC - [Compressional wave](https://en.wikipedia.org/wiki/Longitudinal_wave) travel time: This measure the how long a compressional wave takes to travel through the formationation
#     - RDEP - [Resistivity](https://en.wikipedia.org/wiki/Resistivity_logging) Deep: Electrical resistivity through the rock with a deep penetration
#     - RMED - Resistivity Medium: Electrical resistivity through the rock with a nedium penetration
#     - *Many other well logs were removed as they were not present in all wells*
#     
# Interpreting lithology from well logs is a very hard problem for machine learning because:
#
# - It's usually done by expert humans (Petrophysicists) with years to decades of experience, not an random human
# - it takes into account context in the form of prior knowledge, geology, nearby wells, rock samples, and many more. Many of these are forms of information the machine doesn't have access to
# - The data is unbalanced with important rocks like sandstone sometimes appearing as very this layers
#
#
# <table>
#     <tr>
#         <td>
# <img width="480" src="../../reports/figures/30-4_1.png"/>
#         </td>
#         <td>
# <img width="320" src="../../data/processed/geolink_norge_dataset/location of geolink wells.png"/>
#         </td>
#     </tr>
# </table>
#
#
# ### Data Disclaimer
#
# All the data serving as an input to these notebooks was generously donated by GEOLINK  
# and is CC-by-SA 4.0 
#
# If you use this data please reference the dataset properly to give them credit for their contribution.
#
# **Note:** download data from https://drive.google.com/drive/folders/1EgDN57LDuvlZAwr5-eHWB5CTJ7K9HpDP
#
# Credit to this repo: https://github.com/LukasMosser/geolink_dataset
#
# ### Data Preparation
#
# The geolink dataset we will use in this notebook has been preprocessed. You can find the process of preparation of this dataset in [Data Preparation](../z00_Data_prep/00-mc-prep_geolink_norge_dataset.ipynb)
#
# ## Load Dataset

# +
interim_locations = Path("../../data/processed/geolink_norge_dataset/")
# Load processed dataset
geolink = pd.read_parquet(
    interim_locations / "geolink_norge_well_logs_train.parquet"
).set_index(["Well", "DEPT"])
# Add Depth as column
geolink['DEPT'] = geolink.index.get_level_values(1)

# Work with one well
geolink = geolink.xs("30_4-1")
geolink
# -

geolink['LITHOLOGY_GEOLINK'].astype('category')
geolink['LITHOLOGY_GEOLINK'] = geolink['LITHOLOGY_GEOLINK'].values.remove_unused_categories()

from deep_ml_curriculum.visualization.well_log import plot_facies, plot_well
plot_well("30_4-1", geolink, facies=geolink['LITHOLOGY_GEOLINK'].astype('category').values)

# we only take the data from CALI  onward for X.
X = geolink.iloc[:, 1:]
# LITHOLOGY_GEOLINK will be our class y.
y = geolink["LITHOLOGY_GEOLINK"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2020
)

# ## 3. K-Nearest Neighbors (KNN) <a name="knn"></a>
# **Example of decision boundaries using KNN:**
# <br/>
# <div>
#     Original Dataset. The dataset which consists of three classes (red,green and blue points). (For KNN classification example.)
# <img src="https://upload.wikimedia.org/wikipedia/commons/c/cc/Data3classes.png" width=300 height=300/>
#     Decision Boundaries using 1NN algorithm
# <img src="https://upload.wikimedia.org/wikipedia/commons/5/52/Map1NN.png" width=300 height=300/>
# </div>
#
# Images sources: [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#/media/File:Data3classes.png)
#  distributed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) license.
#           
# The code presented in this section is inspired from the official documentation [here](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py).
#
# According to Wikipedia:
# >k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation. Since this algorithm relies on distance for classification, normalizing the training data can improve its accuracy dramatically.

# Get types of Lithology
classes = list(geolink["LITHOLOGY_GEOLINK"].unique())
print(f"Classes: {classes}")
print(f"Total Classes: {len(classes)}")

# Let's check the classes
y_train.to_numpy()

# There a total of 11 classes. However, like in the majority of machine learning algorithms, it is recommended to encode target values.
#
#
# There are different encoder preprocessing functions available in the scikit learn library. For this example we will use the [<code>LabelEncoder</code>](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html):
#
# > Encode target labels with a value between 0 and n_classes-1. This transformer should be used to encode target values, i.e. y, and not the input X.

# Label Encoder
le = preprocessing.LabelEncoder()
# This will help the LabelEncoder to map the classes to a corresponding value between 0 and n_classes-1
le.fit(classes)

# The number of `neighbors` is a hyperparameter that must be set for this algorithm. We will arbitrarily select a value of 15 for this hyperparameter.
#
# > In machine learning, a hyperparameter is a parameter whose value is used to control the learning process. By contrast, the values of other parameters (typically node weights) are derived via training.
#
# Source: [Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)#:~:text=In%20machine%20learning%2C%20a%20hyperparameter,weights%20are%20derived%20via%20training.)

# Number of Neighbors around our datapoint to be classified
n_neighbors = 15
knn_classifier = KNeighborsClassifier(n_neighbors)
# Now we will use the y values and transform the labels.
transformed_y_train = le.transform(y_train.to_numpy())
print(transformed_y_train)
# Let's fit the data
knn_classifier.fit(X_train, transformed_y_train)

# Evaluation Time
y_pred = knn_classifier.predict(X_test)
y_true = le.transform(y_test.to_numpy())
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")

# We were able to predict the 11 different classes with 93.2% accuracy using new data. This is slightly better than random, however the accuracy is still very low. Let's train using the same algorithm and hyperparameters but this time we will normalise the data.

# ## Normalised Data
#
#
# ### Why normalise the data?
#
# Many machine learning (ML) algorithms attempt to find trends in the data by comparing the features of data points. However, machine learning algorithms usually struggle more in the training phase when the features are on different scales. Normalise the data will almost always improve the results for ML algorithms.
#
# There are different ways to normalised the data and usually, each method has pros and cons. For example, some methods are better at dealing with outliers than others.
#
# Two of the most popular methods for normalising the data are:
#
# * Standard Score:
# $\frac {X-\mu }{\sigma }$
#
# * Min-Max Feature Scaling:
# $X'={\frac {X-X_{\min }}{X_{\max }-X_{\min }}}$
#
# More information about normalisation [here](https://en.wikipedia.org/wiki/Normalization_(statistics))

# Let's use Standard Score for this example
normalized_df = (X - X.mean()) / X.std()

# These are the original max values of the dataset
X.max()

# These are the original max values after normalising
normalized_df.max()

# Let's used the normalized data for both for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    normalized_df, y.to_numpy(), test_size=0.2, random_state=2020
)

# Label Encoder
le = preprocessing.LabelEncoder()
# This will help the LabelEncoder to map the classes to a corresponding value between 0 and n_classes-1
le.fit(y.to_numpy())

# Normalized knn
knn_classifier_norm = KNeighborsClassifier(n_neighbors)
# Now we will use the y values and transform the labels.
transformed_y_train = le.transform(y_train)
# Let's fit the data
knn_classifier_norm.fit(X_train, transformed_y_train)

# Evaluation Time
y_pred = knn_classifier_norm.predict(X_test)
y_true = le.transform(y_test)
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")

# So now we got an accuracy of 90%. Normalising will usually help in the training process. So it's a good practice to preprocess the data before the training phase. However, some machine learning algorithms such as KNN are robust enough to work well with different scales so normalization might not be necessary in some cases. 

# #### So... What if we change the hyperparameters of a classifier such as the number of `neighbors`?
#
# We can see in the examples below how changing <code>n_neighbors</code> affect the performance of the final model.
#
# The accuracy for the same normalised model are:
#
# - n_neighbors = 10, accuracy: 90.9%
# - n_neighbors = 5,  accuracy: 91.6%
# - n_neighbors = 1,  accuracy: 92.2%
# We could  manually try different values until we find the best hyperparameters. Of course, this approach could be very time consuming, in particular, when we have a big feature space and want to optimise many hyperparameters. One way to solve this problem is automating the search of hyperparameters. This is also called Hyperparameter Optimisation. We will get deeper into this concept in the next sessions.

n_neighbors = 10
# Normalized knn
knn_classifier_norm = KNeighborsClassifier(n_neighbors)
# Now we will use the y values and transform the labels.
transformed_y_train = le.transform(y_train)
# Let's fit the data
knn_classifier_norm.fit(X_train, transformed_y_train)
# Evaluation Time
y_pred = knn_classifier_norm.predict(X_test)
y_true = le.transform(y_test)
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")

n_neighbors = 5
# Normalized knn
knn_classifier_norm = KNeighborsClassifier(n_neighbors)
# Now we will use the y values and transform the labels.
transformed_y_train = le.transform(y_train)
# Let's fit the data
knn_classifier_norm.fit(X_train, transformed_y_train)
# Evaluation Time
y_pred = knn_classifier_norm.predict(X_test)
y_true = le.transform(y_test)
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")

n_neighbors = 1
# Normalized knn
knn_classifier_norm = KNeighborsClassifier(n_neighbors)
# Now we will use the y values and transform the labels.
transformed_y_train = le.transform(y_train)
# Let's fit the data
knn_classifier_norm.fit(X_train, transformed_y_train)
# Evaluation Time
y_pred = knn_classifier_norm.predict(X_test)
y_true = le.transform(y_test)
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")

# <div class="alert alert-success">
#   <h2>Exercise 1</h2>
#
#   Let's practice some of the key concepts we have learned so far. 
#     
#   0. Choose the data for well log '30_6-14'. 
#   1. Normalised and split the dataset provided (Use 85\% of the data for training and 15\% for testing)
#   2. Transform the data using label encoding (This is done automatically by scikit-learn for some ML methods. However, it is good to get used to this concept).
#   3. Train a KNN model using <code>n_neighbors</code>= 1,5,10,15
#   4. Compare the accuracy of the different models
#
#   <details>
#   <summary><b>→ Hints</b></summary>
#    - In this notebook we have already covered all the steps necessary to complete this exercise. You need to load again the dataset but selecting this time the well log '30_6-14'. Copy and paste the code and modify it as needed.
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
#     # Exercise
#
#     # Write your code below:
#
#     # 0. Select a different well
#     interim_locations = Path("../../data/processed/geolink_norge_dataset/")
#     # Load processed dataset
#     geolink = pd.read_parquet(
#         interim_locations / "geolink_norge_well_logs_train.parquet"
#     ).set_index(["Well", "DEPT"])
#     # Add Depth as column
#     geolink['DEPT'] = geolink.index.get_level_values(1)
#     # Work with one well
#     geolink = geolink.xs('30_6-14')
#
#     # 1. Normalise dataset and Split well log dataset here
#
#     # We only take the data from CALI  onward for X.
#     X = geolink.iloc[:, 1:]
#     # LITHOLOGY_GEOLINK will be our class y.
#     y = geolink["LITHOLOGY_GEOLINK"]
#     # Normalize data
#     normalized_X = (X - X.mean()) / X.std()
#     # Let's used the normalized data for both for training and testing
#     X_train, X_test, y_train, y_test = train_test_split(
#         normalized_X, y.to_numpy(), test_size=0.15, random_state=1
#     )
#
#     # 2. Transform the data using label encoding
#
#     # Label Encoder
#     le = preprocessing.LabelEncoder()
#     classes = list(geolink["LITHOLOGY_GEOLINK"].unique())
#     # This will help the LabelEncoder to map the classes to a corresponding value between 0 and n_classes-1
#     le.fit(classes)
#     # Now we will use the y values and transform the labels.
#     transformed_y_train = le.transform(y_train)
#     y_test_true = le.transform(y_test)
#     # 3. Train your models 15NN
#
#     # Note: In this example code we are just showing 15NN
#     # Number of Neighbors around our datapoint to be classified
#
#     # 1NN
#     n_neighbors = 1
#     cls_1nn = KNeighborsClassifier(n_neighbors)
#     # Let's fit the data
#     cls_1nn .fit(X_train, transformed_y_train)
#
#     # 5NN
#     n_neighbors = 5
#     cls_5nn = KNeighborsClassifier(n_neighbors)
#     # Let's fit the data
#     cls_5nn .fit(X_train, transformed_y_train)
#
#     # 10NN
#     n_neighbors = 10
#     cls_10nn = KNeighborsClassifier(n_neighbors)
#     # Let's fit the data
#     cls_10nn .fit(X_train, transformed_y_train)
#
#     # 15NN
#     n_neighbors = 15
#     cls_15nn = KNeighborsClassifier(n_neighbors)
#     # Let's fit the data
#     cls_15nn .fit(X_train, transformed_y_train)
#
#     # 4. Compare the accuracy of your models
#
#     # Evaluation Time
#     y_pred = cls_1nn.predict(X_test)
#     print(f"Accuracy 1NN: {accuracy_score(y_test_true, y_pred)}")
#
#     # Evaluation Time
#     y_pred = cls_5nn.predict(X_test)
#     print(f"Accuracy 5NN: {accuracy_score(y_test_true, y_pred)}")
#
#     # Evaluation Time
#     y_pred = cls_10nn.predict(X_test)
#     print(f"Accuracy 10NN: {accuracy_score(y_test_true, y_pred)}")
#
#     # Evaluation Time
#     y_pred = cls_15nn.predict(X_test)
#     print(f"Accuracy 15NN: {accuracy_score(y_test_true, y_pred)}")     
#   ```
#
#   </details>
#
#   </div>
#

# +
# Exercise

# Write your code below:

# 0. Select a different well

# 1. Normalise dataset and Split well log dataset here

# 2. Transform the data using label encoding

# 3. Train your models 1NN, 5NN, 10NN, 15NN

# 4. Compare the accuracy of your models
# -

# # 4. Support Vector Machines (SVM) <a name="svm"></a>
#
# **A bit of history:**
#
# > The Support Vector Machine (SVM) algorithm is a popular machine learning tool that offers solutions for both classification and regression problems. Developed at AT&T Bell Laboratories by Vapnik with colleagues (Boser et al., 1992, Guyon et al., 1993, Vapnik et al., 1997), it presents one of the most robust prediction methods, based on the statistical learning framework.
#
# **How does it works?**
# > An SVM performs classification tasks by constructing hyperplanes in a multidimensional space that separates cases of different class labels. You can use an SVM when your data has exactly two classes, e.g. binary classification problems
#
#
# The graphic below shows how a support vector machine would choose a separating hyperplane for two classes of points in 2D. H1 does not separate the classes. H2 does, but only with a small margin. H3 separates them with the maximum margin.
#
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Svm_separating_hyperplanes_%28SVG%29.svg/1024px-Svm_separating_hyperplanes_%28SVG%29.svg.png" width=300 height=300/>
#
# [Source Image](https://en.wikipedia.org/wiki/Support_vector_machine#/media/File:Svm_separating_hyperplanes_(SVG).svg)
#
# One of the limitations of SVMs is that they inherently do binary classification. However, there are different methods to extend SVMs and use them in multiclass problems. The most common methods involve transforming the problem into a set of binary classification problems, by one of two strategies:
#
# - One vs. the rest. For 𝑘 classes, 𝑘 binary classifiers are trained. Each determines whether an example belongs to its 'own' class versus any other class. The classifier with the largest output is taken to be the class of the example.
# - One vs. one. A binary classifier is trained for each pair of classes. A voting procedure is used to combine the outputs.
#
# Sources: [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine), [SVMs](https://nlp.stanford.edu/IR-book/html/htmledition/multiclass-svms-1.html), [Multiclass SVM](https://stats.stackexchange.com/questions/215342/can-a-linear-svm-only-have-2-classes)
#
#
# **Scikit-learn Implementation:**
#
# Multiclass and Multilabel algorithms:
#
# > **Warning:** All classifiers in scikit-learn do multiclass classification out-of-the-box. You don’t need to use the sklearn.multiclass module unless you want to experiment with different multiclass strategies.
#
# More information about Multiclass classification in scikit-learn [here](https://scikit-learn.org/stable/modules/multiclass.html)
#
# The SVC implementation is not efficient for large datasets. Instead we will use the LinearSVC implementation. Find more information in the [official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
#
# > The implementation is based on libsvm. The fit time scales at least quadratically with the number of samples and might be impractical beyond tens of thousands of samples. For large datasets consider using sklearn.svm.LinearSVC or sklearn.linear_model.SGDClassifier instead, possibly after a sklearn.kernel_approximation.Nystroem transformer.
#
# Sometimes, we will want to train and test our algorithm in a sample fo the data instead of the completed dataset. Some ML methods do not deal well with large datasets so taking a sample from the dataset would be preferable.
#

# +
interim_locations = Path("../../data/processed/geolink_norge_dataset/")
# Load processed dataset
geolink = pd.read_parquet(
    interim_locations / "geolink_norge_well_logs_train.parquet"
).set_index(["Well", "DEPT"])
# Add Depth as column
geolink['DEPT'] = geolink.index.get_level_values(1)

# Work with one well
geolink = geolink.xs("30_4-1")
# -

print("Total rows in Geolink dataset for the selected well:", len(geolink))

# Some algorithms will take significantly more time to train than others. In the case of SVMs the process of training can get slower as the number of data points is increased.
#
# Note: For the example below, we will sample the data to get only 10,000 data points from the original geolink dataset instead of 22,921. This reduction in datapoints may affect the accuracy of the model. But it will be done for demonstration purposes.

# +
# In our case, we will take a sample of 10,000 data points from one well
sample_dataset = geolink.sample(n=10000, replace=False, random_state=2020)

X_sample = sample_dataset.iloc[:, 1:]
y_sample = sample_dataset["LITHOLOGY_GEOLINK"]
# Normalise data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_sample, test_size=0.3, random_state=2020
)

clf = LinearSVC(
    penalty="l2",
    loss="squared_hinge",
    dual=True,
    tol=0.0001,
    C=100,
    multi_class="ovr",
    fit_intercept=True,
    intercept_scaling=1,
    class_weight=None,
    verbose=0,
    random_state=2020,
    max_iter=10000,
)
clf.fit(X_train, y_train)

categories = y_sample.unique()
# Evaluation Time
y_pred = clf.predict(X_test)
y_pred = pd.Categorical(y_pred, categories=categories)
y_true = y_test
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
# -

print(sklearn.metrics.classification_report(y_true, y_pred))

# +
# Plot the results as well logs
X_sample = geolink.iloc[:, 1:]
y_true = geolink["LITHOLOGY_GEOLINK"]
X_scaled = scaler.fit_transform(X_sample)
y_pred = clf.predict(X_scaled)

true = pd.Categorical(y_true)
pred = pd.Categorical(y_pred, categories=true.categories)

f, ax = plt.subplots(nrows=1, ncols=2, figsize=(4, 12))
plot_facies(pred, ax=ax[0], colorbar=False, xlabel='Pred')
plot_facies(true, ax=ax[1], xlabel='True')
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[1].set_yticklabels([])
# -

# The results could be better with a deeper network and more data. We will revit this again in the Recurrent Neural Networks notebook.
#
# We could also:
# - give it a sequence of a logs, so it can see the context up an down the well
# - give it the X and Y location of the well
# - look at more logs, while using the GPU
# - provide some human examples for each well
# - project a geological model or geological maps onto the wells

# # 5. Decision Trees <a name="decision-trees"></a>

#
# > A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements. 
# Decision trees are commonly used in operations research, specifically in decision analysis, to help identify a strategy most likely to reach a goal, but are also a popular tool in machine learning.
#
# Some **advantages** of using decision trees are:
#
# - Simple to understand and interpret: People can understand decision tree models after a brief explanation. Trees can also be displayed graphically in a way that is easy for non-experts to interpret.
#
# - Able to handle both numerical and categorical data. [1]
#
# - Requires little data preparation: Other techniques often require data normalization. Since trees can handle qualitative predictors, there is no need to create dummy variables. [1]
#
# - Performs well with large datasets. Large amounts of data can be analyzed using standard computing resources in a reasonable time.
#
# **Limitations:** 
# - Trees can be very non-robust. A small change in the training data can result in a large change in the tree and consequently the final predictions. [1]
# - Decision-tree learners can create over-complex trees that do not generalize well from the training data.
#
#
# Source: [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree)
#
# Source: [1] Gareth, James; Witten, Daniela; Hastie, Trevor; Tibshirani, Robert (2015). An Introduction to Statistical Learning. New York: Springer. pp. 315. ISBN 978-1-4614-7137-0.
#
# **Note:** Decision trees can be used for classification and regression.

# In the first example, we will train a decision tree on the iris dataset.
#
# We can use the function <code>plot_tree</code> to show the flowchart of the trained model. Find more information about plotting the decision surface of a decision tree [here](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html#sphx-glr-auto-examples-tree-plot-iris-dtc-py). As can be seen in the next plot, the maximum [depth](http://typeocaml.com/2014/11/26/height-depth-and-level-of-a-tree/) of the tree is 5. Usually decision trees with more levels (deeper trees) will be able to discriminate better and take more finetuned decisions, however, deeper trees are also likely to [overfit](https://en.wikipedia.org/wiki/Overfitting) easily. 
#
# <div class="alert alert-info" style="font-size:100%">
#
# **NOTE:** <br>
#
# Use the hyperparameters `max_depth` and `max_features` carefully. Incresing those parameters might increase the [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) of the model but it is also likely to overfit the models at some point.
#     <br/>
# Using the correct metrics of evaluation is key to correctly assess the model.
#
# </div>

# +
# Split dataset geolink. test_size =0.25
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2020
)

clf = tree.DecisionTreeClassifier()
fig, ax = plt.subplots(figsize=(15, 10))
# X0 and y0 from iris_dataset
tree.plot_tree(clf.fit(X_train, y_train), max_depth=3, fontsize=10)
plt.show()
# -

# Now let's train a decision tree (DT) with our geolink dataset. For the next example, we will set the hyperparater max_depth to 100. Notice how the accuracy in the model changes. 

# +
clf = tree.DecisionTreeClassifier(random_state=2020)
clf = clf.fit(X_train, y_train)

# Evaluation Time
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# -

# We got an accuracy of 93%~ just with the default hyperparameters. Let's train the DT again with a different hyperparameter.

# +
clf2 = tree.DecisionTreeClassifier(random_state=2020, max_depth=10)
clf2 = clf2.fit(X_train, y_train)

# Evaluation Time
y_pred = clf2.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# -

# We got now an accuracy of 92%~ just with the hyperparameter max_depth=10. Let's train the DT again with a different hyperparameter. In appearance, the first model would be better, however, there are other metrics besides accuracy that should be taken into account. There are also other methods to avoid overfitting. We will go deeper into this topic in the next sessions.

# # 6. Random Forest <a name="random-forest"></a>
#
# > A random forest is a meta estimator that fits several decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.
#
#
# Random Forest is an ensemble method:
# >The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm to improve generalizability/robustness over a single estimator.
#
# We will learn more about ensemble methods in the next session.
#
# Scikit-learn support Random Forest for classification and regression (<code>RandomForestClassifier</code> and <code>RandomForestRegressor</code>)
#
# > In random forests (see RandomForestClassifier and RandomForestRegressor classes), each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set. Furthermore, when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of size max_features. (See the parameter tuning guidelines for more details). 
# The purpose of these two sources of randomness is to decrease the variance of the forest estimator. Indeed, individual decision trees typically exhibit high variance and tend to overfit. The injected randomness in forests yields decision trees with somewhat decoupled prediction errors. By taking an average of those predictions, some errors can cancel out. Random forests achieve a reduced variance by combining diverse trees, sometimes at the cost of a slight increase in bias. In practice, the variance reduction is often significant hence yielding an overall better model.
#
# This implementation of RandomForest only accepts numerical values. You should always encode categorical values.
#
# Source: [Official Documentation Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
#
# **Note:** The next 2 pieces of code might take several minutes to complete the training.

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Let's train again the model with different hyperparameters.

clf = RandomForestClassifier(max_depth=100, n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# ## References and further reading
# The following sources have been used in the creation of this notebook:
# - [Decision Trees](https://en.wikipedia.org/wiki/Decision_tree)
# - [SVMs](https://nlp.stanford.edu/IR-book/html/htmledition/multiclass-svms-1.html)
# - [Normalisation](https://en.wikipedia.org/wiki/Normalization_(statistics))
# - [Seaborn Datasets](https://seaborn.pydata.org/generated/seaborn.load_dataset.html)
# - [Scikit-learn Datasets](https://scikit-learn.org/stable/datasets/index.html)
#
