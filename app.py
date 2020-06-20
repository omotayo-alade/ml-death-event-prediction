import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model.pkl', 'rb'))

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
    result = model.predict(final_features)
    
    if int(result)== 1: 
        prediction ='Patient died during the follow-up period.'
    else: 
        prediction ='Patient did not die during the follow-up period.'

    return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.debug = True
    app.run()