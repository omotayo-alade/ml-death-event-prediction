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
# df = pd.read_csv('data/processed/processed_data.csv')
df = pd.read_csv('../../data/processed/processed_data.csv')
df.head()

df.columns = ['Age', 'Anaemia', 'CP', 'Diabetes', 'EF', 'HBP',
              'Platelets','SC', 'SS', 'Gender', 'Smoking', 'Time', 'Death_Event']

df.corr()['Death_Event'][:-1].sort_values(ascending=False)

import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.target import FeatureCorrelation

x = df.drop('Death_Event', axis=1)
y = df['Death_Event']
visual = FeatureCorrelation(method='pearson', label=x.columns, sort=True).fit(x,y)
visual.poof();

fig, ax = plt.subplots(figsize=(15,8))
sns.heatmap(df.corr(), annot=True, square=False, ax=ax)
plt.show()

df.head(2)

df['Anaemia'].replace({0:'False', 1:'True'}, inplace=True)
df['Diabetes'].replace({0:'False', 1:'True'}, inplace=True)
df['HBP'].replace({0:'False', 1:'True'}, inplace=True)
df['Gender'].replace({0:'Female', 1:'Male'}, inplace=True)
df['Smoking'].replace({0:'False', 1:'True'}, inplace=True)
df['Death_Event'].replace({0:'Survived', 1:'Died'}, inplace=True)

bins = np.linspace(df['Age'].min(), df['Age'].max(), 5, dtype=('int'))
labels = ['40-53', '53-67', '67-81', '81-95']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
df['Age_Group'].unique()

import seaborn as sns
import matplotlib.pyplot as plt

age_grouped = df.groupby('Age_Group')[['CP', 'EF', 'Platelets', 'SC', 'SS', 'Time']].mean()
age_grouped

anaemia_grouped = df.groupby('Anaemia')[['CP', 'EF', 'Platelets', 'SC', 'SS', 'Time']].mean()
anaemia_grouped

diabetes_grouped = df.groupby('Diabetes')[['CP', 'EF', 'Platelets', 'SC', 'SS', 'Time']].mean()
diabetes_grouped

HBP_grouped = df.groupby('HBP')[['CP', 'EF', 'Platelets', 'SC', 'SS', 'Time']].mean()
HBP_grouped

sex_grouped = df.groupby('Gender')[['CP', 'EF', 'Platelets', 'SC', 'SS', 'Time']].mean()
sex_grouped

smoking_grouped = df.groupby('Smoking')[['CP', 'EF', 'Platelets', 'SC', 'SS', 'Time']].mean()
smoking_grouped

table = pd.crosstab(df['Age_Group'], df['Anaemia'])
table.plot(kind='bar')

table = pd.crosstab(df['Age_Group'], df['Diabetes'])
table.plot(kind='bar')

table = pd.crosstab(df['Age_Group'], df['HBP'])
table.plot(kind='bar')

table = pd.crosstab(df['Age_Group'], df['Gender'])
table.plot(kind='bar')

table = pd.crosstab(df['Age_Group'], df['Smoking'])
table.plot(kind='bar')

table = pd.crosstab(df['Age_Group'], df['Death_Event'])
table.plot(kind='bar')

table = pd.crosstab(df['Anaemia'], df['Death_Event'])
table.plot(kind='bar')

table = pd.crosstab(df['Diabetes'], df['Death_Event'])
table.plot(kind='bar')

table = pd.crosstab(df['HBP'], df['Death_Event'])
table.plot(kind='bar')

table = pd.crosstab(df['Gender'], df['Death_Event'])
table.plot(kind='bar')

table = pd.crosstab(df['Smoking'], df['Death_Event'])
table.plot(kind='bar')

sns.barplot(df['Age_Group'], df['Time'], errwidth=0)

pd.crosstab(df['Age_Group'], df['Death_Event'])


