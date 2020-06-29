import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

model = pickle.load(open('outputs/models/model.pkl', 'rb'))

app = Flask(__name__, template_folder='template') #Initialize the flask App

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    result = model.predict_proba(final_features).tolist()
    output = round((result[0][1] * 100), 3)

    model_prediction = 'Patient has about {}% chance of survival.'.format(output)
    
    return render_template('index.html', prediction=model_prediction, show_prediction=True)

if __name__ == '__main__':
    app.run(debug=True)