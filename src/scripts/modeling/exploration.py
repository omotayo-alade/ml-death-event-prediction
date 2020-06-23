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

df.corr()['death_event'][:-1].sort_values(ascending=False)

import seaborn as sns
import matplotlib.pyplot as plt
from yellowbrick.target import FeatureCorrelation

x = df.drop('death_event', axis=1)
y = df['death_event']
visual = FeatureCorrelation(method='pearson', label=x.columns, sort=True).fit(x,y)
visual.poof();

df.head(2)

df['anaemia'].replace({0:'False', 1:'True'}, inplace=True)
df['diabetes'].replace({0:'False', 1:'True'}, inplace=True)
df['high_blood_pressure'].replace({0:'False', 1:'True'}, inplace=True)
df['sex'].replace({0:'Female', 1:'Male'}, inplace=True)
df['smoking'].replace({0:'False', 1:'True'}, inplace=True)

bins = np.linspace(df['age'].min(), df['age'].max(), 5, dtype=('int'))
labels = ['40-53', '53-67', '67-81', '81-95']
df['age_binned'] = pd.cut(df['age'], bins=bins, labels=labels, include_lowest=True)
df['age_binned'].unique()

age_grouped = df.groupby('age_binned')[['creatinine_phosphokinase',
                                    'ejection_fraction', 'platelets',
                                   'serum_creatinine', 'serum_sodium','time']].sum()
age_grouped

anaemia_grouped = df.groupby('anaemia')[['creatinine_phosphokinase',
                                   'ejection_fraction', 'platelets',
                                   'serum_creatinine', 'serum_sodium', 'time']].mean()
anaemia_grouped.head()

diabetes_grouped = df.groupby('diabetes')[['creatinine_phosphokinase',
                                   'ejection_fraction', 'platelets',
                                   'serum_creatinine', 'serum_sodium', 'time']].mean()
diabetes_grouped.head()

bloodP_grouped = df.groupby('high_blood_pressure')[['creatinine_phosphokinase',
                                   'ejection_fraction', 'platelets',
                                   'serum_creatinine', 'serum_sodium', 'time']].mean()
bloodP_grouped.head()

sex_grouped = df.groupby('sex')[['creatinine_phosphokinase',
                                   'ejection_fraction', 'platelets',
                                   'serum_creatinine', 'serum_sodium', 'time']].mean()
sex_grouped.head()

smoking_grouped = df.groupby('smoking')[['creatinine_phosphokinase',
                                   'ejection_fraction', 'platelets',
                                   'serum_creatinine', 'serum_sodium', 'time']].mean()
smoking_grouped.head()

table = pd.crosstab(df['age_binned'], df['anaemia'])
table.head()

table = pd.crosstab(df['age_binned'], df['diabetes'])
table.head()

table = pd.crosstab(df['age_binned'], df['high_blood_pressure'])
table.head()

table = pd.crosstab(df['age_binned'], df['sex'])
table.head()

table = pd.crosstab(df['age_binned'], df['smoking'])
table.head()

table = pd.crosstab(df['age_binned'], df['death_event'])
table.head()

df.shape




