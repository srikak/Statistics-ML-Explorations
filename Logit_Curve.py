# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:55:06 2024

@author: ksrik
"""

#%% Load libraries
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

# Define the logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))

# Define the logit function
def logit(p):
    return np.log(p / (1 - p))

# Generate values for x in the range -10 to 10
x_values = np.linspace(-10, 10, 400)

# Calculate the logistic function values
logistic_values = logistic(x_values)
# Calculate the logit function values
logit_values = logit(logistic_values)

# Create traces for logistic function
logistic_trace = go.Scatter(
    x=x_values,
    y=logistic_values,
    mode='lines',
    name='Logistic Function',
    line=dict(color='blue')
)

# Create traces for logit function
logit_trace = go.Scatter(
    x=logistic_values,
    y=logit_values,
    mode='lines',
    name='Logit Function',
    line=dict(color='orange')
)

# Create subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=('Logistic Function (Sigmoid)', 'Logit Function'))

# Add traces to subplots
fig.add_trace(logistic_trace, row=1, col=1)
fig.add_trace(logit_trace, row=1, col=2)

# Update layout
fig.update_layout(
    title='Logistic and Logit Functions',
    showlegend=True
)

# Update x-axis and y-axis titles for each subplot
fig.update_xaxes(title_text='x', row=1, col=1)
fig.update_yaxes(title_text='σ(x)', row=1, col=1)

fig.update_xaxes(title_text='p', row=1, col=2)
fig.update_yaxes(title_text='logit(p)', row=1, col=2)

# Show the plot
fig.show()