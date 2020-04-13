# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 19:03:19 2020

@author: 1426391

K-Means Clustering
"""

import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs

"""################## k-Means on a randomly generated dataset
Lets create our own dataset for this lab!

First we need to set up a random seed. Use numpy's random.seed() function, 
where the seed will be set to 0"""

np.random.seed(0)

"""Next we will be making <i> random clusters </i> of points by using the <b> make_blobs </b> class. The <b> make_blobs </b> class can take in many inputs, but we will be using these specific ones. <br> <br>
<b> <u> Input </u> </b>
<ul>
    <li> <b>n_samples</b>: The total number of points equally divided among clusters. </li>
    <ul> <li> Value will be: 5000 </li> </ul>
    <li> <b>centers</b>: The number of centers to generate, or the fixed center locations. </li>
    <ul> <li> Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]] </li> </ul>
    <li> <b>cluster_std</b>: The standard deviation of the clusters. </li>
    <ul> <li> Value will be: 0.9 </li> </ul>
</ul>
<br>
<b> <u> Output </u> </b>
<ul>
    <li> <b>X</b>: Array of shape [n_samples, n_features]. (Feature Matrix)</li>
    <ul> <li> The generated samples. </li> </ul> 
    <li> <b>y</b>: Array of shape [n_samples]. (Response Vector)</li>
    <ul> <li> The integer labels for cluster membership of each sample. </li> </ul>
</ul>
"""
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

# Display the scatter plot of the randomly generated data.

plt.scatter(X[:, 0], X[:, 1], marker='.')

## Setting up K-Means
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

# Now let's fit the KMeans model with the feature matrix we created above, X
k_means.fit(X)
