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

import pandas as pd
import numpy as np

# Load data into data frame
# df = pd.read_csv('data/processed/processed_data.csv')
df = pd.read_csv('../../data/processed/processed_data.csv')
df.sample(10)

# Import libraries for modeling processes
import sklearn
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# Separate features and target variables
X = df.drop('Death_Event', axis=1)
Y = df['Death_Event']


# Define parameter tunning function for Logistic Regression
def get_params_LR(x, y):
    params = {'solver':['liblinear', 'saga'],
              'penalty':['l2', 'l1'], 'C':[x for x in range(1, 11)],
              'max_iter':[200, 100, 400,300]}
    estimator = LogisticRegression()
    GS = GridSearchCV(estimator, params, scoring='accuracy', cv=5)
    GS.fit(x, y)
    solver = GS.best_params_['solver']
    penalty = GS.best_params_['penalty']
    C = GS.best_params_['C']
    max_iter = GS.best_params_['max_iter']
    return GS.best_params_ #[solver, penalty, C, max_iter]


# Define parameter tunning function for Decision Tree Classifier
def get_params_DT(x, y):
    params = {'criterion':['entropy', 'gini'],
              'max_depth':[1,2,3,4,5,6,7,8,9],
              'ccp_alpha':[x * 0.05 for x in range(1, 10)],}
    estimator = DecisionTreeClassifier()
    GS = GridSearchCV(estimator, params, scoring='accuracy', cv=5)
    GS.fit(x, y)
    criterion = GS.best_params_['criterion']
    max_depth = GS.best_params_['max_depth']
    ccp_alpha = GS.best_params_['ccp_alpha']
    return GS.best_params_ #[criterion, max_depth, ccp_alpha]


# Define parameter tunning function for Support Vector Classifier
def get_params_SVC(x, y):
    params = {'C':[1,2,3,4,5],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
    estimator = SVC()
    GS = GridSearchCV(estimator, params, scoring='accuracy', cv=5)
    GS.fit(x, y)
    C = GS.best_params_['C']
    kernel = GS.best_params_['kernel']
    return GS.best_params_ #[C, kernel]


# Create model building function
def build_model(alg, x, y, processing=None):
    np.random.seed(25)
    
    # Select feature using recursive feature selection
    model = DecisionTreeClassifier(max_depth=4)
    fit = RFE(model, n_features_to_select=4).fit(x, y)
    feature_rank = pd.DataFrame({'Feature':X.columns,
                                 'Rank':fit.ranking_,
                                 'Selected':fit.support_})
    feature_rank = feature_rank.sort_values(by='Rank', ascending=True)
    RFE_selected_features = feature_rank[feature_rank['Selected'] == True]
    RFE_features = x[RFE_selected_features['Feature']]
    x = RFE_features
    
    if processing is not None:
        x = processing.fit_transform(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    if alg == 'lreg':
        param = get_params_LR(x_train, y_train)
        alg = LogisticRegression(solver=param['solver'],
                                 penalty=param['penalty'],
                                 C=param['C'],
                                 max_iter=param['max_iter'])
    
    if alg == 'dtree':
        param = get_params_DT(x_train, y_train)
        alg = DecisionTreeClassifier(criterion=param['criterion'],
                                     max_depth=param['max_depth'],
                                     ccp_alpha=param['ccp_alpha'])
    
    if alg == 'svc':
        param = get_params_SVC(x_train, y_train)
        alg = SVC(C=param['C'],
                  kernel=param['kernel'])

    model = alg.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    return {
            'training score': model.score(x_train, y_train),
            'recall score': recall_score(y_test, y_pred),
            'precision score': precision_score(y_test, y_pred),
            'accuracy score': accuracy_score(y_test, y_pred),
            'parameter used': param,
            'feature rank': feature_rank,
            'feature used': list(RFE_features.columns)}


pd.set_option('max_colwidth', None)
pd.set_option('display.max_columns', None)

# Create dictionary for results of different models
result_dict = {}

# Fit Logistic Regression model
result_dict['logistic_regression'] = build_model('lreg', X, Y, processing=PowerTransformer())

# Fit Decision Tree Classifier Model
result_dict['decision_tree_classifier'] = build_model('dtree', X, Y, processing=PowerTransformer())

# Fit Support Vector Classifier model
result_dict['support_vector_classifier'] = build_model('svc', X, Y, processing=PowerTransformer())

result_dict['logistic_regression']['feature rank'].transpose()

# Display results
result_df = pd.DataFrame(result_dict)
result_df.head().transpose()

# <p>Seeing as Logistic Regression gave the best relative result, it is chosen for the final model</p>


