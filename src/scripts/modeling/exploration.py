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

# Load data into data frame
# df = pd.read_csv('data/raw/raw_data.csv')
df = pd.read_csv('../../data/raw/raw_data.csv')

df.columns

# Rename columns
df.columns = ['Age', 'Anaemia', 'CP', 'Diabetes', 'EF', 'HBP',
              'Platelets', 'SC', 'SS', 'Gender', 'Smoking', 'Time', 'Death_Event']

df.shape

df.info()

df.describe().transpose()

df['Age'] = df['Age'].astype('int64')

# Export data to csv
df.to_csv('../../data/processed/processed_data.csv', index=False)

df.corr()['Death_Event'][:-1].sort_values(ascending=False)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams as rc
from yellowbrick.target import FeatureCorrelation

rc['xtick.labelsize'] = 15.0
rc['ytick.labelsize'] = 15.0
rc['xtick.direction'] = 'out'
rc['axes.labelsize'] = 15.0
rc['axes.titlesize'] = 18.0
rc['savefig.format'] = 'png'
rc['savefig.dpi'] = 600
rc['legend.fontsize'] = 15

x = df.drop('Death_Event', axis=1)
y = df['Death_Event']
fig = plt.figure(figsize=(8,6))
corr = FeatureCorrelation(method='pearson', label=x.columns, sort=True).fit(x,y);
plt.savefig('../../outputs/visuals/correlations')
corr.show();

fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(df.corr(), annot=True, square=False, ax=ax);
ax.set_title('Correlations between features')
plt.savefig('../../outputs/visuals/correlations_all')
plt.show()

# Age distribution of Patients
fig, ax = plt.subplots(figsize=(8,6))
sns.kdeplot(df['Age'], legend=False, shade=True, ax=ax);
plt.savefig('../../outputs/visuals/age_distribution')
ax.set_title('Age Distribution of Patients')


