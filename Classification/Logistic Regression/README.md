
# Logistic Regression with Python
In this notebook, you will learn Logistic Regression, and then, you'll create a model for a telecommunication company, to predict when its customers will leave for a competitor, so that they can take some action to retain the customers.


## What is different between Linear and Logistic Regression?
While Linear Regression is suited for estimating continuous values (e.g. estimating house price), it is not the best tool for predicting the class of an observed data point. In order to estimate the class of a data point, we need some sort of guidance on what would be the most probable class for that data point. For this, we use Logistic Regression.

## Recall linear regression:

As you know, __Linear regression__ finds a function that relates a continuous dependent variable, _y_, to some predictors (independent variables _x1_, _x2_, etc.). For example, Simple linear regression assumes a function of the form:

𝑦=𝜃0+𝜃1∗𝑥1+𝜃2∗𝑥2+...
 

and finds the values of parameters _θ0_, _θ1_, _𝜃2_, etc, where the term _𝜃0_ is the "intercept". It can be generally shown as:

ℎθ(𝑥)=𝜃𝑇𝑋
 
Logistic Regression is a variation of Linear Regression, useful when the observed dependent variable, y, is categorical. It produces a formula that predicts the probability of the class label as a function of the independent variables.

Logistic regression fits a special s-shaped curve by taking the linear regression and transforming the numeric estimate into a probability with the following function, which is called sigmoid function 𝜎:

ℎθ(𝑥)=𝜎(θ𝑇𝑋)=𝑒(θ0+θ1∗𝑥1+θ2∗𝑥2+...)1+𝑒(θ0+θ1∗𝑥1+θ2∗𝑥2+...)
 
Or:
𝑃𝑟𝑜𝑏𝑎𝑏𝑖𝑙𝑖𝑡𝑦𝑂𝑓𝑎𝐶𝑙𝑎𝑠𝑠1=𝑃(𝑌=1|𝑋)=𝜎(θ𝑇𝑋)=𝑒θ𝑇𝑋1+𝑒θ𝑇𝑋
 
In this equation,  θ𝑇𝑋  is the regression result (the sum of the variables weighted by the coefficients), exp is the exponential function and  𝜎(θ𝑇𝑋)  is the sigmoid or logistic function, also called logistic curve. It is a common "S" shape (sigmoid curve).

So, briefly, Logistic Regression passes the input through the logistic/sigmoid but then treats the result as a probability:


The objective of Logistic Regression algorithm, is to find the best parameters θ, for ℎ_θ(𝑥) = 𝜎({θ^TX}), in such a way that the model best predicts the class of each case.

Customer churn with Logistic Regression
A telecommunications company is concerned about the number of customers leaving their land-line business for cable competitors. They need to understand who is leaving. Imagine that you’re an analyst at this company and you have to find out who is leaving and why.

Lets first import required libraries:

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
%matplotlib inline 
import matplotlib.pyplot as plt

## About dataset
We’ll use a telecommunications data for predicting customer churn. This is a historical customer data where each row represents one customer. The data is relatively easy to understand, and you may uncover insights you can use immediately. Typically it’s less expensive to keep customers than acquire new ones, so the focus of this analysis is to predict the customers who will stay with the company.

This data set provides info to help you predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs.

## The data set includes information about:

### Customers who left within the last month – the column is called Churn

### Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies

### Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges

### Demographic info about customers – gender, age range, and if they have partners and dependents

Load the Telco Churn data
Telco Churn is a hypothetical data file that concerns a telecommunications company's efforts to reduce turnover in its customer base. Each case corresponds to a separate customer and it records various demographic and service usage information. Before you can work with the data, you must use the URL to get the ChurnData.csv.

