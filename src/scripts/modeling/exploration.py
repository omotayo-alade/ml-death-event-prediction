# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Import dataframe libraries
import pandas as pd
import numpy as np

# Load dataset into dataframe
df = pd.read_csv('../../data/processed/processed_data.csv')
df.head()

# Drop Unnamed column
df.drop('Unnamed: 0', axis=1, inplace=True)
df.head()

df.corr()['death_event'][:-1].sort_values(ascending=False)

import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.target import FeatureCorrelation

x = df.drop('death_event', axis=1)
y = df['death_event']
visual = FeatureCorrelation(method='pearson', label=x.columns, sort=True).fit(x,y)
visual.poof();

# + active=""
# df.plot(kind='scatter', x='age', y='ejection_fraction', c='serum_sodium', cmap='viridis')

# + active=""
# table = pd.crosstab(df['smoking'], df['diabetes'])
# table.plot(kind='bar', stacked=True)
# plt.ylabel('Frequency Distribution')
# -

df.head(2)

# + active=""
# next thing to do is groupby
