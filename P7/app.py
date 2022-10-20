import json
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

app = Flask(__name__)

test_df = pd.read_csv("files/P7_test_df.csv")
test_df = test_df.drop(columns=['Unnamed: 0'])

@app.route("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.route("/labels")
def labels():
    Customer_ID = pd.read_csv("files/Customer_ID.csv")
    Customer_ID = Customer_ID.drop(columns=['Unnamed: 0'])
    return json.loads(Customer_ID.to_json())

@app.route("/customer")
def data():
    selector = request.args.get("selector") 
    selector = eval(selector)
    Selected_Customer = test_df.loc[test_df['SK_ID_CURR'] == selector]
    Selected_Customer.to_csv("files/selection.csv")
    return json.loads(Selected_Customer.to_json())

@app.route("/local")
def local():

  #Local Features Case for a chosen Selected Customer

    Selected_Customer = pd.read_csv("files/selection.csv")
    Selected_Customer = Selected_Customer.drop(columns=['Unnamed: 0'])

    feature_list = list(test_df.columns)

    X = test_df.drop(['TARGET'], axis=1).values
    y = test_df['TARGET'].values

    data = Selected_Customer.drop(['TARGET'], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    undersample = RandomUnderSampler(sampling_strategy=1)
    X_train, y_train = undersample.fit_resample(X_train, y_train)
    rfc = RandomForestClassifier(max_depth=13, min_samples_leaf=2, min_samples_split=8, n_estimators=552)
    score = rfc.fit(X_train, y_train).predict(data)
  
    Credit_given_test = np.max(rfc.predict_proba(data))

    if score==0:
      credit_score=Credit_given_test

    else:
      credit_score=(1-Credit_given_test)

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

  #print('Customer Score Ready')
    return json.loads(ten_most_important_df.to_json())


@app.route('/global_data/')

def global_data():

  feature_list = list(test_df.columns)

  X = test_df.drop(['TARGET'], axis=1).values
  y = test_df['TARGET'].values

  filename = 'files/final_model.sav'
  loaded_model = pickle.load(open(filename, 'rb'))
  
  # Get numerical feature importances
  importances = list(loaded_model.feature_importances_)

  # List of tuples with variable and importance
  feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

  # Sort the feature importances by most important first
  feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

  #Ten most important features
  ten_most_important = feature_importances[0:10]

  Global_Features = pd.DataFrame(ten_most_important)

  Global_Features.columns = ['Feature', 'Importance']

  print('Global Features Ready')

  # Print out the feature and importances 
  return json.loads(Global_Features.to_json())

if __name__ == "__main__":
    app.run(debug=True)