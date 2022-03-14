# DISASTER RESPONSE NLP MODEL

## Introduction

The primary aims of disaster response are rescue from immediate danger and stabilization of the physical and emotional condition of survivors. These go hand in hand with the recovery of the dead and the restoration of essential services such as water and power. 

In this Project, we'll work with data set containing real messages that were sent during disaster events. This dataset will help in building a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

# File Descriptions

* ETL_Pipeline_Preparation.ipynb: *Steps Performed in Jupyter notebook to Prepare the Data (Extracts,Transform and Load)
* ML_Pipeline_Preparation.ipynb: *Steps Performed in Jupyter notebook to Prepare the ML Pipeline.
* process_data.py: *Python File to run ETL pipeline that cleans data and stores in database
* train_classifier.py: *Python File to run ML pipeline that trains classifier and saves 
* run.py: *Python File to run the Web App 

# Web App Snippet
![Web App](/DisasterResponse_ML/Web_app.JPG)


# Libraries Used

  1. Nltk
  2. Pandas
  3. Scikit-Learn/sklearn
  4. Pickle
  5. re
  6. sqlite3
  7. plotly
  8. JSON
  9. flask

# Acknowledgements
I would like to give credit to:
* ![Udacity](https://classroom.udacity.com/nanodegrees/nd025)
* ![ML Algos](https://machinelearningmastery.com/)
