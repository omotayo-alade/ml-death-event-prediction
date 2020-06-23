# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
    
# Loading dataset
dataset = pd.read_csv('data/processed/processed_data.csv')

x = dataset[['creatinine_phosphokinase', 'ejection_fraction', 'serum_creatinine', 'time']]
y = dataset['death_event']

x = PowerTransformer().fit_transform(x)

# Splitting dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)

# Creating model
model = LogisticRegression(solver='saga', penalty='l2', C=0.2, max_iter=200, random_state=25)

# Fitting the model with training data
classifier = model.fit(x_train, y_train)

# Saving model to disk
pickle.dump(classifier, open('outputs/models/model.pkl','wb'))