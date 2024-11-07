# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:18:14 2024

@author: ksrik
"""
#%% Load Libraries
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

#%%
# Set seed for reproducibility
np.random.seed(42)

# Example 1
n = 1000  # Number of observations
p = 5000  # Number of predictors
real_p = 15  # Number of true predictors

# Generate the data
X = np.random.randn(n, p)
y = np.sum(X[:, :real_p], axis=1) + np.random.randn(n)

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Ridge Regression (alpha=0)
#ridge = RidgeCV(alphas=np.logspace(-6, 6, 13), cv=10)
ridge = Ridge(alpha = 0, max_iter = 100)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

ridge_score = ridge.score(X_test, y_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)

# Lasso Regression (alpha=1)
#lasso = LassoCV(alphas=np.logspace(-6, 6, 13), cv=10)
lasso = Lasso(alpha = 1, max_iter = 100)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)

lasso_score = lasso.score(X_test, y_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)

# Elastic Net Regression (alpha=0.5)
#elastic_net = ElasticNetCV(alphas=np.logspace(-6, 6, 13), l1_ratio=0.5, cv=10)
elastic_net = ElasticNet(alpha = 0.5, max_iter = 100)
elastic_net.fit(X_train, y_train)
elastic_net_pred = elastic_net.predict(X_test)

elastic_net_score = elastic_net.score(X_test, y_test)
elastic_net_mse = mean_squared_error(y_test, elastic_net_pred)

# Plot MSE for different models
model_names = ['Ridge Regression', 'Lasso Regression', 'Elastic Net (alpha=0.5)']
mse_values = [ridge_mse, lasso_mse, elastic_net_mse]

fig1 = go.Figure()
fig1.add_trace(go.Bar(x=model_names, y=mse_values, marker_color='royalblue'))
fig1.update_layout(
    title='MSE for Different Regression Models',
    xaxis_title='Model',
    yaxis_title='Mean Squared Error (MSE)',
    xaxis=dict(tickangle=-45)
)
fig1.show()

# Try different alphas for Elastic Net
results = []
for alpha in np.arange(0, 1.1, 0.1):
    elastic_net = ElasticNet(alpha = alpha, max_iter = 100)
    elastic_net.fit(X_train, y_train)
    elastic_net_pred = elastic_net.predict(X_test)
    mse = mean_squared_error(y_test, elastic_net_pred)
    results.append({'alpha': alpha, 'mse': mse})

results_df = pd.DataFrame(results)

# Plot MSE vs. Alpha for Elastic Net
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=results_df['alpha'], y=results_df['mse'], mode='lines+markers', marker_color='red'))
fig2.update_layout(
    title='MSE vs. Alpha for Elastic Net Regression',
    xaxis_title='Alpha (L1 Ratio)',
    yaxis_title='Mean Squared Error (MSE)',
    xaxis=dict(tickmode='linear')
)
fig2.show()