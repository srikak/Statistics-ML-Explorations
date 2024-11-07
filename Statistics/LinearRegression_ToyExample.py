# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:22:55 2024

@author: ksrik
"""

#%% Load libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

#%% Load data
data = {
    'Size': [1500, 1600, 1700, 1800, 1900, 2000],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000]
}
df = pd.DataFrame(data)

#%% Linear Regression Model
X = df[['Size']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

#%% Plots
# Scatter plot for actual data
fig = px.scatter(df, x="Size", y="Price", title="House Prices vs Size", labels={"Size": "Size of House (sq ft)", "Price": "Price (USD)"})

# Add the regression line
# Generate predictions across the entire range of X for visualization purposes
X_range = pd.DataFrame({'Size': np.linspace(X['Size'].min(), X['Size'].max(), 100)})
y_range_pred = model.predict(X_range)

# Add the regression line trace
regression_line = go.Scatter(x=X_range['Size'], y=y_range_pred, mode='lines', name='Regression Line', line=dict(color='red'))
fig.add_trace(regression_line)

# Show the plot
fig.show()
