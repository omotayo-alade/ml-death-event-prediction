# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Loading dataset
dataset = pd.read_csv('processed_data.csv')

x = dataset[['creatinine_phosphokinase', 'ejection_fraction', 'serum_creatinine', 'time']]
y = dataset['death_event']

# Splitting dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Creating pipeline
steps = [('transformer', PowerTransformer()),
         ('model', LogisticRegression(solver='liblinear', penalty='l2', C=1.0, max_iter=200, random_state=0))]

# Fitting the model with training data
classifier = Pipeline(steps).fit(x_train, y_train)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))