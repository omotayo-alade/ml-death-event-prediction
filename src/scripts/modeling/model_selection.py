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

pd.set_option('max_colwidth', None)
pd.set_option('display.max_columns', None)

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
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline

# Separate features and target variables
X = df.drop('Death_Event', axis=1)
Y = df['Death_Event']


# Define parameter tunning function for Logistic Regression
def get_params_LR(x, y):
    params = {'solver':['liblinear', 'saga'],
              'penalty':['l2', 'l1'],
              'C':[round(x, 1) for x in np.arange(0.1, 1, 0.1)],
              'max_iter':[200, 100, 400,300]}
    estimator = LogisticRegression()
    GS = GridSearchCV(estimator, params, scoring='recall', cv=4, n_jobs=-1)
    GS.fit(x, y)
    return GS.best_params_ #[solver, penalty, C, max_iter]


# Define parameter tunning function for Decision Tree Classifier
def get_params_DT(x, y):
    params = {'criterion':['entropy', 'gini'],
              'max_depth':[1,2,3,4,5,6,7,8,9],
              'ccp_alpha':[x * 0.05 for x in range(1, 10)]}
    estimator = DecisionTreeClassifier()
    GS = GridSearchCV(estimator, params, scoring='recall', cv=4, n_jobs=-1)
    GS.fit(x, y)
    return GS.best_params_ #[criterion, max_depth, ccp_alpha]


# Define parameter tunning function for Support Vector Classifier
def get_params_SVC(x, y):
    params = {'C':[round(x, 1) for x in np.arange(0.1, 1, 0.1)],
              'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
    estimator = SVC()
    GS = GridSearchCV(estimator, params, scoring='recall', cv=4, n_jobs=-1)
    GS.fit(x, y)
    return GS.best_params_ #[C, kernel]


# Create model building function for Train Test
def build_model(alg, x, y, processing=None):
    np.random.seed(25)
    
    # Select feature using recursive feature selection
    model = DecisionTreeClassifier(max_depth=5)
    fit = RFE(model, n_features_to_select=4).fit(x, y)
    feature_rank = pd.DataFrame({'Feature':X.columns,
                                 'Rank':fit.ranking_,
                                 'Selected':fit.support_})
    feature_rank = feature_rank.sort_values(by='Rank', ascending=True)
    RFE_selected_features = feature_rank[feature_rank['Selected'] == True]
    RFE_features = x[RFE_selected_features['Feature']]
    x = RFE_features
    
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
        alg = SVC(C=param['C'], kernel=param['kernel'])

    pipe = Pipeline([('scaler', processing), ('estimator', alg)])
    model = pipe.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    return {
            'training score': model.score(x_train, y_train),
            'roc_auc score': roc_auc_score(y_test, y_pred),
            'recall score': recall_score(y_test, y_pred),
            'precision score': precision_score(y_test, y_pred),
            'accuracy score': accuracy_score(y_test, y_pred),
            'parameter used': param,
            'feature used': list(RFE_features.columns)}


# Create dictionary for results of different models
result_dict = {}

# Fit Logistic Regression model
result_dict['logistic_regression'] = build_model('lreg', X, Y, processing=MinMaxScaler())

# Fit Decision Tree Classifier Model
result_dict['decision_tree_classifier'] = build_model('dtree', X, Y, processing=MinMaxScaler())

# Fit Support Vector Classifier model
result_dict['support_vector_classifier'] = build_model('svc', X, Y, processing=MinMaxScaler())

# Display results
result_df = pd.DataFrame(result_dict)
result_df.transpose()


# Create model building function for KFold
def build_KFold_model(alg, x, y, processing=None):
    np.random.seed(25)
    
    # Select feature using recursive feature selection
    model = DecisionTreeClassifier(max_depth=5)
    fit = RFE(model, n_features_to_select=4).fit(x, y)
    feature_rank = pd.DataFrame({'Feature':X.columns,
                                 'Rank':fit.ranking_,
                                 'Selected':fit.support_})
    feature_rank = feature_rank.sort_values(by='Rank', ascending=True)
    RFE_selected_features = feature_rank[feature_rank['Selected'] == True]
    RFE_features = x[RFE_selected_features['Feature']]
    x = RFE_features
    
    if alg == 'lreg':
        param = get_params_LR(x, y)
        alg = LogisticRegression(solver=param['solver'],
                                 penalty=param['penalty'],
                                 C=param['C'],
                                 max_iter=param['max_iter'])
    
    if alg == 'dtree':
        param = get_params_DT(x, y)
        alg = DecisionTreeClassifier(criterion=param['criterion'],
                                     max_depth=param['max_depth'],
                                     ccp_alpha=param['ccp_alpha'])
    
    if alg == 'svc':
        param = get_params_SVC(x, y)
        alg = SVC(C=param['C'], kernel=param['kernel'])

    pipe = Pipeline([('scaler', processing), ('estimator', alg)])
    all_score = cross_val_score(pipe, x, y, scoring='recall', cv=5, n_jobs=-1)
    score = all_score.mean()
    
    return {'score': score,
            'parameter used': param,
            'feature rank': feature_rank,
            'feature used': list(RFE_features.columns)}


# Create dictionary for results of different models
result_dict = {}

# Fit Logistic Regression model
result_dict['logistic_regression'] = build_KFold_model('lreg', X, Y, processing=PowerTransformer())

# Fit Decision Tree Classifier Model
result_dict['decision_tree_classifier'] = build_KFold_model('dtree', X, Y, processing=PowerTransformer())

# Fit Support Vector Classifier model
result_dict['support_vector_classifier'] = build_KFold_model('svc', X, Y, processing=PowerTransformer())

# Display results
result_df = pd.DataFrame(result_dict)
result_df.transpose()[['score', 'parameter used', 'feature used']]

X = X[['CP', 'EF', 'SC', 'Time']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=25)
estimator = LogisticRegression(C=0.4, max_iter=300, penalty='elasticnet', solver='saga', l1_ratio=1, random_state=25)
pipe = Pipeline([('scaler', MinMaxScaler()), ('est', estimator)])
model = pipe.fit(x_train, y_train)
y_pred = model.predict(x_test)
yhat = model.predict(X)
prob = model.predict_proba(X)
print(yhat[0:5])
print(prob[0:5])
print('training score: \t', model.score(x_train, y_train))
print('roc_auc score:  \t', roc_auc_score(y_test, y_pred))
print('recall score:   \t', recall_score(y_test, y_pred))
print('precision score: \t', precision_score(y_test, y_pred))
print('accuracy score: \t', accuracy_score(y_test, y_pred))

X = X[['CP', 'EF', 'SC', 'Time']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=25)
estimator = SVC(C=0.4, kernel='linear', probability=True, random_state=25)
pipe = Pipeline([('scaler', MinMaxScaler()), ('est', estimator)])
model = pipe.fit(x_train, y_train)
y_pred = model.predict(x_test)
yhat = model.predict(X)
prob = model.predict_proba(X)
print(yhat[0:5])
print(prob[0:5])
print('training score: \t', model.score(x_train, y_train))
print('roc_auc score:  \t', roc_auc_score(y_test, y_pred))
print('recall score:   \t', recall_score(y_test, y_pred))
print('precision score: \t', precision_score(y_test, y_pred))
print('accuracy score: \t', accuracy_score(y_test, y_pred))


