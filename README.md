# ML Classification Model for Predicting Death Event following Heart Failure

This is a machine learning classification model that predicts whether a patient died during follow-up period (days) after having a heart failure. The model was developed based on a recent dataset donated to University of California, Irvine [UCI](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) Machine Learning repository on 5th Feb, 2020.

## Table of Contents
* [Installation](#Installation)
* [Project Structure](#Structure)
* [Running the project](#Running)
* [Licensing, Authors, Acknowledgements](#Licensing)

## Installation <a name="Installation"></a>
The scripts require Python versions of 3.7, Jupyter Notebook, Scikit Learn (for ML model ), Pickle (for creating a serialized version of the model) Flask (for API), all available through pip and anaconda packages.

## Project Structure <a name="Structure"></a>
The project contains the following components:
* data - This directory contains raw and processed dataset from which the model is built.
* output - This directory contains a subdirectory named models where the serialized model is saved.
* src - This directory contains two main directory:
  * ingest: this conatins app.py, a Flask API script that receives patients' clinical features through GUI or API calls, computes the precited value based on our model and           returns it. It also contains to sub-directories, (a) static, where css stylesheet is stored and (b) template, where index.html for user feature imputation is stored.
  * modeling: this contains model.py script used for buiding the final model and model_selection.py script used for model_selection.
* data_description.txt - This gives meaning to each clinical feature contained in the dataset.

## Running the Project <a name="Running"></a>
You could run the project both online and locally. If you choose to run online, visit https://death-event-prediction.herokuapp.com

If you prefer to run locally, follow steps below:

* Ensure that you are in the project home directory. Create the machine learning model by running below command from command prompt
```
python src/scripts/modeling/model.py
```
This would create a serialized version of the model into a file model.pkl

* Run app.py using below command to start Flask API
```
python src/scripts/ingest/app.py
```
By default, flask will run on port 5000.

* If browser does not open automatically, navigate to URL http://127.0.0.1:5000/ (or) http://localhost:5000

You should be able to view the homepage.

Enter valid integer values in input boxes and hit Predict.

If everything goes well, you should  be able to see the predcited salary vaule on the HTML page!
check the output here: http://127.0.0.1:5000/predict

## Licensing, Authors, Acknowledgements <a name="Licensing"></a>
The data used in this machine learning model is from University of California, Irvine [UCI](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) Machine Learning repository. Credit goes to [Maajid Khan](https://github.com/MaajidKhan/DeployMLModel-Flask) whose API, HTML and CSS scripts, with few modifications, were used for deploying the model.
