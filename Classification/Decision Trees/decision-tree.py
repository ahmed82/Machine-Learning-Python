# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 09:12:16 2020

@author: 1426391

Decision Trees

When building a decision tree, we want to split the nodes in a way that 
    Decreases entropy 
        and 
    Increases information gain.
"""

import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# !wget -O drug200.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv
my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]

"""
Pre-processing
Using my_data as the Drug.csv data read by pandas, declare the following variables:

X as the Feature Matrix (data of my_data)
<li> <b> y </b> as the <b> response vector (target) </b> </li>
Remove the column containing the target name since it doesn't contain numeric values.
"""

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

"""As you may figure out, some featurs in this dataset are catergorical such as Sex or BP. 
Unfortunately, Sklearn Decision Trees do not handle categorical variables. 
But still we can convert these features to numerical values. 
pandas.get_dummies() Convert categorical variable into dummy/indicator variables."""

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# Target variable.
y = my_data["Drug"]
y[0:5]

""" ############################### Setting up the Decision Tree ##########################
We will be using train/test split on our decision tree.
import train_test_split from sklearn.cross_validation."""
from sklearn.model_selection import train_test_split

"""  train_test_split will return 4 different parameters. We will name them:
X_trainset, X_testset, y_trainset, y_testset

The train_test_split will need the parameters:
X, y, test_size=0.3, and random_state=3.

The X and y are the arrays required before the split, the test_size represents the ratio of the testing dataset, and the random_state ensures that we obtain the same splits."""

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

X_trainset.shape
y_trainset.shape

X_testset.shape
y_testset.shape

""" ##################### Modeling #############
Create an instance of the DecisionTreeClassifier = drugTree.
Inside of the classifier, specify criterion="entropy" 
so we can see the information gain of each node."""

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters

""" fit the data with the training feature 
        matrix X_trainset and training response vector y_trainset"""
drugTree.fit(X_trainset,y_trainset)

""" ##------------------- Prediction -----------------## 
Make some predictions on the testing dataset and store it into a variable called predTree."""

predTree = drugTree.predict(X_testset)

# You can print out predTree and y_testset if you want to visually 
# compare the prediction to the actual values.
print (predTree [0:5])
print (y_testset [0:5])

""" # -------------------- Evaluation ---------------- ##
 Import metrics from sklearn and check the accuracy of our model."""

from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

"""
Accuracy classification score computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.

In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

## Practice ##
Can you calculate the accuracy score without sklearn ?
"""

""" ---------------- Visualization -------------"""
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
# %matplotlib inline


dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')











