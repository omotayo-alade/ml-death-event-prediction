# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/modeling//py
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

import pandas as pd
import numpy as np

# Load data into data frame
df = pd.read_csv('data/raw/raw_data.csv')

df.head()

df.columns

# Rename columns
df.columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
       'death_event']

df.shape

df.info()

df.describe().transpose()

df['age'] = df['age'].astype('int64')

df.to_csv('data/processed/processed_data.csv')

# Import libraries for visualzation
import seaborn as sns
import matplotlib.pyplot as plt

# +
# Visualize distribution of features
fig, ax = plt.subplots(figsize=(20, 15))

df.plot(kind='density', subplots=True, layout=(4,4), sharex=False, sharey=False, ax=ax)

plt.show()

# +
# Visualize correlation of features
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(df.corr(), annot=True, ax=ax)

plt.show()
# -

# Import libraries for modeling processes
import sklearn
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Separate features and target variables
X = df.drop('death_event', axis=1)
Y = df['death_event']


# Define parameter tunning function for Logistic Regression
def get_params_LR(x, y):
    params = {'solver':['liblinear', 'saga'],
              'penalty':['l2', 'l1'], 'C':[1.0,1.5,2.0,2.5],
              'max_iter':[200, 100, 400,300],
              'random_state':list(range(10))}
    estimator = LogisticRegression()
    GS = GridSearchCV(estimator, params, scoring='accuracy', cv=10)
    GS.fit(x, y)
    solver = GS.best_params_['solver']
    penalty = GS.best_params_['penalty']
    C = GS.best_params_['C']
    max_iter = GS.best_params_['max_iter']
    random_state = GS.best_params_['random_state']
    return [solver, penalty, C, max_iter, random_state]


# Define parameter tunning function for Decision Tree Classifier
def get_params_DT(x, y):
    params = {'criterion':['entropy', 'gini'],
              'max_depth':[1,2,3,4,5,6,7,8,9],
              'ccp_alpha':[0.1, 0.2, 0.5, 1.0,1.5,2.0,2.5],
              'random_state':list(range(10))}
    estimator = DecisionTreeClassifier()
    GS = GridSearchCV(estimator, params, scoring='accuracy', cv=10)
    GS.fit(x, y)
    criterion = GS.best_params_['criterion']
    max_depth = GS.best_params_['max_depth']
    ccp_alpha = GS.best_params_['ccp_alpha']
    random_state = GS.best_params_['random_state']
    return [criterion, max_depth, ccp_alpha, random_state]


# Define parameter tunning function for Support Vector Classifier
def get_params_SVC(x, y):
    params = {'C':[1.0,1.5,2.0,2.5],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
              'random_state':list(range(10))}
    estimator = SVC()
    GS = GridSearchCV(estimator, params, scoring='accuracy', cv=10)
    GS.fit(x, y)
    C = GS.best_params_['C']
    kernel = GS.best_params_['kernel']
    random_state = GS.best_params_['random_state']
    return [C, kernel, random_state]


# Create model building function
def build_model(alg, x, y, processing=None):
    
    # Perform Recursive Feature Elimination for feature selection
    model = DecisionTreeClassifier(max_depth=4)
    fit = RFE(model, n_features_to_select=4).fit(x, y)
    feature_rank = pd.DataFrame({'Feature':x.columns,
                                 'Rank':fit.ranking_,
                                 'Selected':fit.support_})
    feature_rank = feature_rank.sort_values(by='Rank', ascending=True)
    RFE_selected_features = feature_rank[feature_rank['Selected'] == True]
    RFE_features = x[RFE_selected_features['Feature']]
    x_new = RFE_features
   
    if processing is not None:
        x_new = processing.fit_transform(x_new)
    
    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.2)
    
    if alg == 'lreg':
        param = get_params_LR(x_train, y_train)
        alg = LogisticRegression(solver=param[0], penalty=param[1], C=param[2], max_iter=param[3], random_state=param[4])
    
    if alg == 'dtree':
        param = get_params_DT(x_train, y_train)
        alg = DecisionTreeClassifier(criterion=param[0], max_depth=param[1], ccp_alpha=param[2], random_state=param[3])
    
    if alg == 'svc':
        param = get_params_SVC(x_train, y_train)
        alg = SVC(C=param[0], kernel=param[1], random_state=param[2])

    model = alg.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    return {
            'training score': model.score(x_train, y_train),
            'accuracy score': accuracy_score(y_test, y_pred),
            'features used': list(RFE_features.columns),
            'parameter used': param
            }


# Create dictionary for results of different models
result_dict = {}


def compare_results(result_dict):
    for key in result_dict:
        print('algorithm: ', key)
        print('parameter used:', result_dict[key]['parameter used'])
        print('features used:', result_dict[key]['features used'])
        print('training score:', result_dict[key]['training score'])
        print('accuracy score:', result_dict[key]['accuracy score'])


# Fit Logistic Regression model
result_dict['logistic_regression'] = build_model('lreg', X, Y, processing=PowerTransformer())

# Fit Decision Tree Classifier Model
result_dict['decision_tree_classifier'] = build_model('dtree', X, Y, processing=PowerTransformer())

# Fit Support Vector Classifier model
result_dict['support_vector_machine'] = build_model('svc', X, Y, processing=PowerTransformer())

# Compare results of the three models
compare_results(result_dict)
