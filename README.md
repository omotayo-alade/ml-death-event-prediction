# ML Classification Model for Predicting Death Event following Heart Failure

This is a machine learning classification model that predicts whether a patient died during follow-up period (days) after having a heart failure. The model was developed based on a recent dataset donated to University of California, Irvine [UCI](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) Machine Learning repository on 5th Feb, 2020.

## Table of Contents
* [Installation](#Installation)
* [Project Structure](#Structure)
* [Running the project](#Running)
* [Licensing, Authors, Acknowledgements](#Licensing)

## Installation <a name="Installation"></a>
The scripts require Python versions of 3.*, Jupyter Notebook, Scikit Learn (for ML model ), Pickle (for creating a serialized version of the model) Flask (for API), all available through pip and anaconda packages.

## Project Structure <a name="Structure"></a>
The project contains the following components:
* model.py - This contains code for the Machine Learning model to predict death event (whether patient died during follow up period after having heart failure) based on training data in 'dataset.csv' file.
* app.py - This contains Flask APIs that receives patients' clinical features through GUI or API calls, computes the precited value based on our model and returns it.
* template - This folder contains the HTML template (index.html) that takes patients' clinical features as input values and displays the prediction whether patient died during the follow-up period or not.
* static - This directory contains the css folder that holds style.css for styling index.html.
* description.txt - This gives meaning to each clinical feature contained in the dataset.
* model_selection.ipynb - This is a Jupyter notebook file used for feature selection and model selection.

## Running the Project <a name="Running"></a>
* Ensure that you are in the project home directory. Create the machine learning model by running below command from command prompt
```
python model.py
```
This would create a serialized version of the model into a file model.pkl

* Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

* If browser does not open automatically, navigate to URL http://127.0.0.1:5000/ (or) http://localhost:5000

You should be able to view the homepage.

Enter valid integer values in input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited salary vaule on the HTML page!
check the output here: http://127.0.0.1:5000/predict

## Licensing, Authors, Acknowledgements <a name="Licensing"></a>
The data used in this machine learning model is from University of California, Irvine [UCI](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) Machine Learning repository. Credit goes to [Maajid Khan](https://github.com/MaajidKhan/DeployMLModel-Flask) whose API, HTML and CSS scripts, with few modifications, were used for deploying the model.
