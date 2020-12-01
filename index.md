---
layout: default
---

# Motivation
---
![Heart Health](/images/heart-health-2.jpg)

> We want to promote general public health by increasing people’s awareness of heart disease factors. Our model can potentially assist physicians to identify people at risk.

# Problem Definition
---
In heart disease, the heart is unable to push the required amount of blood to other parts of the body. The diagnosis of heart disease through traditional medical history has been considered as not reliable in many aspects. To classify the healthy people and people with heart disease, noninvasive-based methods such as machine learning are reliable and efficient.[1]

In this project we are going to predict the heart using different features including age,sex,blood pressure, number of main vessels and so on. To get quite fine results and find correlation between different features.

# Introduction
---
In our project, our goal is to train and create a model to have high accuracy in predicting if patients get heart disease or not. We are interested in this project for the following reason. Doctors need to make sure their analyzed result from the reference model is intensely reliable by different features. The dataset we use contains 76 features, but only 14 of them are related to our project. All data points on the dataset come from About 300 patients in Cleveland. We got our dataset directly from Uci.[2]

# Data Collection
---
At first we want to use the heart disease dataset from kaggle, which is part of the original dataset from Uci. But it is only from cleveland with 300 samples, and this will be too small for us to generate convincing results. So we directly check the original data, it contains four area: Hungary, Virginia, Switzerland and Cleveland. Image of features is given as follows:

![Features](/images/features.png)

However, when integrating these data to the clean Cleveland dataset, we found they have significantly many missing entries. For some sub dataset, we discovered that nearly an entire feature can be missing. We created a “missing data image” where missing data entries are highlighted in yellow. A missing feature is represented as a “vertical streak”. image of missing features is given as follows:

![Missing Features](/images/missing_features.png)
                              
    Images from the left to right are Clevelan, Hungary, Switzerland, Virginia.

We utilized the k-nearest-neighbors technique to clean the integrated dataset. Surprisingly, we found no single completely clean data point from the three other datasets except for the Cleveland dataset, which instead has only clean data. Taking that into consideration, we only used the Cleveland data as the reference to calculate the nearest neighbors and attempted to restore the missing data one by one in the remaining three datasets. For a numerical feature, we took the arithmetic mean of the neighbors, and for a categorical one, we assigned the most frequent value among the neighbors using voting.

Since there are both categorical and numerical features in the dataset, plus different numerical fields can have drastically different ranges. For example, some measurements are in hundreds but some are tiny decimals. Therefore, it is absolutely necessary for us to standardize the data before using any distance based measures like kNN. We were also not sure how categorical (mostly binary) data differs from numerical in calculating distance, so we left a flexible weighing proportion between the two, default to 1:1. In addition, we require a data point to have at least 8 intact features to survive data cleaning. The reason is obvious: a data point with too many missing entries will be unrepresentative no matter what genius cleaning techniques you try, so they will be generally detrimental to our ML models.

This data cleaning is not perfect, but is expected to provide a reasonably large (850+) and representative population for our supervised and unsupervised methods. Though the “column streak” features (with too many missing data points) are eventually recovered, we recommend discarding them in training because if a feature is mostly inferred from others, it will be too dependent on others and thus becoming useless. We will finally compare the performance of different supervised methods on the clean data (Cleveland) with that on recovered data (all four) to see if this data augmentation brings more generalizability.

# Visualization
---
We first use Pearson correlation to generate graph among 13 features (except label), the image is given as follows:

![Missing Features](/images/corr.png)

Then we use different plots to show the relations between single features and target value (label), some of the visualizations are given as follows:

![Missing Features](/images/chest_pain.png)

![Missing Features](/images/cholestrol.png)

![Missing Features](/images/age.png)

Besides, we also draw age and gender distribution of all samples:

![Missing Features](/images/age_distribution.png)


![Missing Features](/images/gender.png)

# Feature Selection
---
We use PCA to select features (from 2 to 13), and use Logistic Regression, KNN, SVM and Kmeans to test the accuracy (training samples is 70% of the total dataset). Results are given as follows: (Orange line shows the accuracy achieved without PCA)

![Missing Features](/images/PCA.png)

In the above plots above we have the reduced dimension of x axis and accuracy of y axis. From the accuracy plot we know that in this problem supervised learning is better than unsupervised learning (Knn is the best), and we should use all the features provided in the datatset instead of doing dimension reduction on it.

# Methods
---
We explore our data and build our model through both unsupervised and supervised methods.[3]

## Unsupervised:
1. PCA: reduce feature dimensionality by identifying the most relevant features and discovering the correlations between them.
2. DBSCAN: group dense clusters and identify outliers to remove before training on a sensitive supervised model.
# Results
---
Kmeans: Apply the elbow method to find the best number of clusters, train the model and plot the graph by number of cluster vs the square distance between each one of clusters, we try different number of clusters here.


## Supervised:
1. Logistic Regression: a perfect algorithm suitable for **binary classification** also as first shot to see if data is **linearly separable**.
2. SVM: apply different **kernel methods and compare** its performance with the logistic regression approach.
3. NN: build a simple multi layer perceptron network for this classification task as an introductory step toward **deep learning**.
4. KNN: a **simple and robust** classification method; fast to compute given the dataset is relatively small with 850~ points
5. Random Forest: we will explore the power of **ensemble learning** models in terms of improving accuracies and **preventing overfitting**.

# Results
---
<!-- Results are shown above in different sections. -->
## Supervised:
1. Random Forest: We experimented with single decision tree as well as
random forest classifiers on our datasets. We tweaked parameters including
the **max_depth** of the trees, and we analyzed the importance of different features.

![Missing Features](/images/dt.png)
![Missing Features](/images/rf.png)
![Missing Features](/images/fi.png)

From the feature importance analysis, we could see that the three most important features that affect heart disease are: The chest pain experienced(*cp*), The person's cholesterol measurement in mg/dl(*chol*), and the slope of the peak exercise ST segment(*slope*).

2. SVM: We experimented with support vector machines with different kernels. The kernel we tested upon including linear kernel, RBF (radial basis function) kernel, polynomial kernel and sigmoid kernel.

![SVM Accuracy](/images/SVM.png)

The resulting accuracy are shown above. We find out that the linear kernel fits the data best, with testing accuracy around 78%.

3. Neural Network: The Network structure is given as follows:

![nn_structure](/images/nn_structure.png)

While the train loss and accuracy are also given. The final accuracy can reach to 84% after 40 iterations.
![nn_loss](/images/nn_loss.png)
![nn_acc](/images/nn_acc.png)

# Discussion
---
During this phase of the project, we have some discussions on data cleaning, visualization, feature selection and some unsupervised && supervised methods. For data cleaning, we examined the columns in the dataset and hold discussion about the best way to fill out the empty entires. For visualization, we each contributed to finding out good ways to display the data by trying different libraries. Then we divided out work into person and perform centern machine learning methods on the data and collect the results. Based on PCA testing results, we decided to use all features for training and testing. We explored methods which are either supervised and unsupervised on the dataset, and obtained results as displayed above. The best accuracy achieved so far is the supervised method of random forest with depth 12, which is around 88%. There might be other ways to further improve the testing result for practical use, but our result obtained so far generates an accuracy over 85% and is good enough for experimental results.


# References
---
[1] Haq A U, Li J P, Memon M H, et al. *A hybrid intelligent system framework for the prediction of heart disease using machine learning algorithms[J]*. Mobile Information Systems, 2018, 2018.

[2] UCI Heart Disease Data Set, https://archive.ics.uci.edu/ml/datasets/heart+disease

[3] Palechor F M, De la Hoz Manotas A, Colpas P A, et al. *Cardiovascular Disease Analysis Using Supervised and Unsupervised Data Mining Techniques[J]*. JSW, 2017, 12(2): 81-90.
