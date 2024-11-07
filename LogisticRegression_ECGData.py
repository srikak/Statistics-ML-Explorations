# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:40:54 2024

@author: ksrik
"""

#%% Import libraries
import requests
from io import StringIO

import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_curve, roc_auc_score

#%% Load data
url = "https://raw.githubusercontent.com/StatQuest/logistic_regression_demo/master/processed.cleveland.data"

response = requests.get(url)

if response.status_code == 200:
    # Load data directly into a pandas DataFrame
    csv_data = response.text
    data = pd.read_csv(StringIO(csv_data), header=None)

    # Set column names
    data.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                   "thalach", "exang", "oldpeak", "slope", "ca", "thal", "hd"]

else:
    print("Failed to download the data. Status code:", response.status_code)

#%% Pre-process data

data.replace("?", np.nan, inplace=True)

# Map the data
data['sex'] = data['sex'].map({0: 'F', 1: 'M'})
#data['hd'] = data['hd'].map({0: 'Healthy', 1: 'Unhealthy'})
data['hd'] = np.where(data['hd'] == 0, 'Healthy', 'Unhealthy')
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')

# Define the type of the column
data['sex'] = data['sex'].astype('category')
data['cp'] = data['cp'].astype('category')
data['fbs'] = data['fbs'].astype('category')
data['restecg'] = data['restecg'].astype('category')
data['exang'] = data['exang'].astype('category')
data['slope'] = data['slope'].astype('category')
data['ca'] = data['ca'].astype('category')
data['thal'] = data['thal'].astype('category')
data['hd'] = data['hd'].astype('category')

# Remove nan rows
data.dropna(inplace=True)

# Cross tabulation

print(pd.crosstab(data['hd'], data['sex']),"\n")
print(pd.crosstab(data['hd'], data['cp']),"\n")
print(pd.crosstab(data['hd'], data['fbs']),"\n")
print(pd.crosstab(data['hd'], data['restecg']),"\n")
print(pd.crosstab(data['hd'], data['exang']),"\n")
print(pd.crosstab(data['hd'], data['slope']),"\n")
print(pd.crosstab(data['hd'], data['ca']),"\n")
print(pd.crosstab(data['hd'], data['thal']),"\n")

#%% Logistic Regression - Single Parameter

X_sex = pd.get_dummies(data['sex'], drop_first=True)
y = data['hd'].cat.codes

logistic_sex = LogisticRegression()
logistic_sex.fit(X_sex, y)

print(f"y = {logistic_sex.intercept_[0]:.2f} + {logistic_sex.coef_[0][0]:2f}x")

ll_null = log_loss(y, [y.mean()] * len(y))
ll_proposed = log_loss(y, logistic_sex.predict_proba(X_sex))

pseudo_r2 = 1 - (ll_proposed / ll_null)

chi2 = 2 * (ll_null - ll_proposed)
p_value = 1 - stats.chi2.cdf(chi2, df=1)

print(f"R-squared = {pseudo_r2:.2f}, p = {p_value:.2f}\n") 

#%% Logistic Regression - Multiple Parameters

data['probability_of_hd'] = logistic_sex.predict_proba(X_sex)[:, 1]

X = pd.get_dummies(data.drop(columns=['hd', 'probability_of_hd']), drop_first=True)
y = data['hd'].cat.codes

logistic_full = LogisticRegression(max_iter=2000)
logistic_full.fit(X, y)

print(f"Intercept = {logistic_full.intercept_}\n")
print(f"Coeffs = {logistic_full.coef_}\n")

ll_null = log_loss(y, [y.mean()] * len(y))
ll_proposed = log_loss(y, logistic_full.predict_proba(X))

pseudo_r2 = 1 - (ll_proposed / ll_null)

chi2 = 2 * (ll_null - ll_proposed)
p_value = 1 - stats.chi2.cdf(chi2, df=len(logistic_full.coef_[0]) - 1)

print(f"R-squared = {pseudo_r2:.2f}, p = {p_value:.2f}") 