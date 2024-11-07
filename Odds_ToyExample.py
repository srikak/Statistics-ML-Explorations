# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:42:30 2024

@author: ksrik
"""

#%% Import libraries
import numpy as np  
import pandas as pd  

# Import plotly for data visualization
import plotly.graph_objects as go  
from plotly.subplots import make_subplots
import plotly.io as pio  
pio.renderers.default = "browser"  # Set the default renderer to browser

#%% Generate sample space for throwing two dice
faces = [1, 2, 3, 4, 5, 6]  # Define the faces of a die

# Create a DataFrame with all combinations of the two dice
sample_space = pd.DataFrame([(d1, d2) for d1 in faces for d2 in faces], 
                            columns=['Die1', 'Die2'])

#%% Query: Odds of getting a sum > n

# Define the threshold sum
n = 2

# Calculate the sum of the two dice
sample_space['Sum'] = sample_space['Die1'] + sample_space['Die2']

# Calculate the number of favourable outcomes
favourable_outcomes = len(sample_space[sample_space['Sum'] > n])

# Calculate the total number of outcomes
total_outcomes = len(sample_space)

# Calculate the probability
probability = favourable_outcomes / total_outcomes

# Calculate the odds
odds = probability / (1 - probability)

# Print the results
if odds >= 1:
    print(f"For sum > {n}, odds = {odds:.2f} --> In favour\n")
elif odds >= 0 and odds < 1:
    print(f"For sum > {n}, odds = {odds:.2f} --> Against\n")

#%% Odds distribution
odds_distribution = []
log_odds = []
outcomes = []

# Loop through possible sums to calculate odds and log odds
for n in range(0, 12):
    favourable_outcomes = len(sample_space[sample_space['Sum'] > n])
    outcomes.append(favourable_outcomes)
    
    p = favourable_outcomes / len(sample_space)
    
    # Handle division by zero
    if p == 1:
        odds_distribution.append(np.inf)
        log_odds.append(np.inf)
    else:
        odds_distribution.append(p / (1 - p))
        log_odds.append(np.log(p / (1 - p)))

#%% Plots
fig = go.Figure()  # Create a new figure

# Create a color array based on the odds distribution
colors = ['green' if o >= 1 else 'red' for o in odds_distribution]

# Add trace for odds distribution
fig.add_trace(go.Scatter(x=outcomes, y=odds_distribution, 
                         mode='lines+markers', 
                         name='Odds', line=dict(color='blue'), 
                         marker=dict(color=colors, size=10)))

# Add trace for log odds distribution
fig.add_trace(go.Scatter(x=outcomes, y=log_odds, 
                         mode='lines+markers', 
                         name='log(Odds)', line=dict(color='yellow'), 
                         marker=dict(color=colors, size=10)))

# Update layout
fig.update_layout(
    title='Odds and log(Odds) for Sums of Two Dice',
    xaxis_title='Sum Greater Than',
    yaxis_title='Value',
    legend_title='Legend'  
)

# Show the figure
fig.show()
