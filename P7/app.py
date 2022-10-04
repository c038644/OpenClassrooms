##########################################################
# to run: FLASK_APP=server.py flask run
##########################################################
import json
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.ensemble import BalancedRandomForestClassifier

app = Flask(__name__)

@app.route('/local/')

def local():

  #Local Features Case for a chosen Selected Customer
  test_df = pd.read_csv(
    "C:/Users/Farida/Documents/Data_Science/P7/Final/files/P7_test_df.csv")

  Selected_Customer = pd.read_csv(
    "C:/Users/Farida/Documents/Data_Science/P7/Final/files/selection.csv")

  print('files loaded')

  Selected_Customer = Selected_Customer.drop(columns=['Unnamed: 0'])
  test_df = test_df.drop(columns=['Unnamed: 0'])

  print(Selected_Customer.shape)
  print(test_df.shape)

  feature_list = list(test_df.columns)

  X = test_df.drop(['TARGET'], axis=1).values
  y = test_df['TARGET'].values

  data = Selected_Customer.drop(['TARGET'], axis=1).values

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

  rfc = BalancedRandomForestClassifier(max_depth=13, min_samples_leaf=2, min_samples_split=8, n_estimators=552, sampling_strategy='all')
  
  print('RFC')

  score = rfc.fit(X_train, y_train).predict(data)

  Credit_given_test = np.max(rfc.predict_proba(data))

  if score==0:
    credit_score=Credit_given_test

  else:
    credit_score=(1-Credit_given_test)

  print('Get importances')

  # Get numerical feature importances
  importances = list(rfc.feature_importances_)

  # List of tuples with variable and importance
  feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

  # Sort the feature importances by most important first
  feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

  #Ten most important features
  ten_most_important = feature_importances[0:10]

  ten_most_important_df = pd.DataFrame(ten_most_important)

  ten_most_important_df.columns = ['Feature', 'Importance']

  ten_most_important_df['Credit Score'] = credit_score

  ten_most_important_df['Credit Granted?'] = None

  if credit_score>=0.35:
    ten_most_important_df['Credit Granted?'] = ten_most_important_df['Credit Granted?'].fillna('Yes')
  elif credit_score>=0.25:
    ten_most_important_df['Credit Granted?'] = ten_most_important_df['Credit Granted?'].fillna('Risky')
  else:
    ten_most_important_df['Credit Granted?'] = ten_most_important_df['Credit Granted?'].fillna('No')

  ten_most_important_df.to_csv(
    "C:/Users/Farida/Documents/Data_Science/P7/Final/files/Customer_score.csv")

  print('Customer Score Ready')
  return json.dumps(ten_most_important_df.to_json())


@app.route('/global_data/')

def global_data():

  test_df = pd.read_csv("C:/Users/Farida/Documents/Data_Science/P7/Final/files/P7_test_df.csv")

  print('files loaded')

  #Global Features Case

  feature_list = list(test_df.columns)

  X = test_df.drop(['TARGET'], axis=1).values
  y = test_df['TARGET'].values

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

  rfc = BalancedRandomForestClassifier(max_depth=13, min_samples_leaf=2, min_samples_split=8, n_estimators=552, sampling_strategy='all')

  print('RFC')

  rfc.fit(X_train, y_train)

  print('Get importances')

  # Get numerical feature importances
  importances = list(rfc.feature_importances_)

  # List of tuples with variable and importance
  feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

  # Sort the feature importances by most important first
  feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

  #Ten most important features
  ten_most_important = feature_importances[0:10]

  Global_Features = pd.DataFrame(ten_most_important)

  Global_Features.columns = ['Feature', 'Importance']

  print('Global Features Ready')

  Global_Features.to_csv("C:/Users/Farida/Documents/Data_Science/P7/Final/files/Global_Features.csv")

  # Print out the feature and importances 
  return json.dumps(Global_Features.to_json())

if __name__ == "__main__":
    app.run(debug=False)