---
layout: default
---

# Motivation
---
![Heart Health](/images/heart-health-2.jpg)

> We want to promote general public health by increasing people’s awareness of heart disease factors. Our model can potentially assist physicians to identify people at risk.

[YouTube Video](https://www.youtube.com/watch?v=k106eh61UK8)

# Problem Definition
---
In heart disease, the heart is unable to push the required amount of blood to other parts of the body. The diagnosis of heart disease through traditional medical history has been considered as not reliable in many aspects. To classify the healthy people and people with heart disease, noninvasive-based methods such as machine learning are reliable and efficient.[1]

In this project we are going to predict the heart using different features including age,sex,blood pressure, number of main vessels and so on. To get quite fine results and find correlation between different features.

# Introduction
---
In our project, our goal is to train and create a model to have high accuracy in predicting if patients get heart disease or not. We are interested in this project for the following reason. Doctors need to make sure their analyzed result from the reference model is intensely reliable by different features. The dataset we use contains 76 features, but only 14 of them are related to our project. All data points on the dataset come from About 300 patients in Cleveland. We got our dataset from Kaggle.

# Methods
---
We explore our data and build our model through both unsupervised and supervised methods.[2]

## Unsupervised:
1. PCA: reduce feature dimensionality by identifying the most relevant features and discovering the correlations between them.
2. DBSCAN: group dense clusters and identify outliers to remove before training on a sensitive supervised model.

## Supervised:
1. Logistic Regression: a perfect algorithm suitable for **binary classification** also as first shot to see if data is **linearly separable**.
2. SVM: apply different **kernel methods and compare** its performance with the logistic regression approach.
3. NN: build a simple multi layer perceptron network for this classification task as an introductory step toward **deep learning**.
4. KNN: a **simple and robust** classification method; fast to compute given the dataset is relatively small with 300~ points
5. Random Forest: we will explore the power of **ensemble learning** models in terms of improving accuracies and **preventing overfitting**.

# Potential Results
---
- Accurate prediction
- Visualization of Data and Training Results

The model we build for this project, both supervised and unsupervised, should make predictions about the probability of a person to get heart disease, based on different data such as age, sex, blood pressure etc. We would also generate the visualization of training results, such as confusion matrices, training loss data, accuracy among different groups of people etc. Besides, we will find out the best method of training as well as the architecture of neural networks for supervised learning. After all, we want to find out the best model to predict heart disease based on the data points provided in the database.

# Discussion
---
During this phase of the project, we have some discussions on datasets. The dataset we chose on Kaggle is just an excerpt from the original dataset, and contains relatively a small amount of data. This could potentially lead to model overfitting. Therefore, we are thinking of including more data from the original datasets, which would require more data cleaning analysis but could lead to better models.[3]

![Dataset First Glance](/images/raw-data-table.png)

# References
---
[1] Haq A U, Li J P, Memon M H, et al. *A hybrid intelligent system framework for the prediction of heart disease using machine learning algorithms[J]*. Mobile Information Systems, 2018, 2018.

[2] Palechor F M, De la Hoz Manotas A, Colpas P A, et al. *Cardiovascular Disease Analysis Using Supervised and Unsupervised Data Mining Techniques[J]*. JSW, 2017, 12(2): 81-90.

[3] UCI Heart Disease Data Set, https://archive.ics.uci.edu/ml/datasets/heart+disease
