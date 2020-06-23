# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
    
# Loading dataset
dataset = pd.read_csv('data/processed/processed_data.csv')

x = dataset[['ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']] 
y = dataset['death_event']

x = PowerTransformer().fit_transform(x)

# Splitting dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Creating model
model = LogisticRegression(solver='liblinear', penalty='l1', C=0.1, max_iter=200)

# Fitting the model with training data
classifier = model.fit(x_train, y_train)

# Saving model to disk
pickle.dump(classifier, open('outputs/models/model.pkl','wb'))