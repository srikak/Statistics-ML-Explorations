# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:09:35 2024

@author: ksrik
"""

import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"

# For .xlsx files
df = pd.read_excel('Data EFC.xlsx')

# For .xls files
# df = pd.read_excel('your_file.xls', engine='xlrd')

# Display the first few rows of the DataFrame
print(df.head())

df = df.dropna()
df = df.reset_index(drop=True)

X = df['Gross fixed capital formation (% of GDP)']
Y = df['GDP Growth']



fig = px.scatter(df, 
                 x = X[258:271], 
                 y = Y[258:271])

fig.show()