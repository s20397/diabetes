"""
This is a boilerplate pipeline 'data_processing_diabetes'
generated using Kedro 0.18.3
"""
from typing import Any, Dict
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import wandb
from pycaret.datasets import get_data
from pycaret.classification import setup
from sklearn.metrics import accuracy_score, roc_auc_score
import yaml

def preprocess_diabetes(diabetes: pd.DataFrame):
    diabetes['Glucose'] = diabetes['Glucose'].replace(0,diabetes['Glucose'].mean())
    diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(0,diabetes['BloodPressure'].mean())
    diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(0,diabetes['SkinThickness'].mean())
    diabetes['Insulin'] = diabetes['Insulin'].replace(0,diabetes['Insulin'].mean())
    diabetes['BMI'] = diabetes['BMI'].replace(0,diabetes['BMI'].mean())
    return diabetes

def split_data(data: pd.DataFrame, parameters: Dict[str, Any]):
    x = data.drop(columns='Outcome')
    y = data.loc[:,'Outcome']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=parameters.get('seed'))
    return x_train,x_test,y_train,y_test

def create_model(df: pd.DataFrame):
    model_setup = setup(data=df, target="Outcome")
    best_model = model_setup.compare_models()
    return best_model

def normalize_features(data):
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data

def train_model(model, x_train, y_train):
    model.fit(x_train,y_train)
    return model

def predict(model, x_test):
    predictions = model.predict(x_test)
    return predictions

def get_classification_report(predictions, y_test):
    report = classification_report(y_test,predictions)
    return report

def get_confussion_matrix(predictions, y_test):
    matrix = confusion_matrix(y_test,predictions)
    return matrix

def evaluate_model(model, x_test, y_test,x_train, y_train):
    # Initiate wandb project
    wandb.init(project="diabetes-2")
    
    labels = y_test.unique()
    y_pred = model.predict(x_test)
    y_probas = model.predict_proba(x_test)[:,1]
    
    # evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probas)
        
    n_estimators = {'min' : 25, 'max' : 200} 
    max_depth = {'min' : 3, 'max' : 10} 
    
    wandb.sklearn.plot_class_proportions(y_train, y_test, labels)
    wandb.sklearn.plot_learning_curve(model, x_train, y_train)
    wandb.sklearn.plot_feature_importances(model)
    
    wandb.log({"n_estimators": n_estimators})
    wandb.log({"max_depth": max_depth})
    wandb.log({"accuracy": accuracy})
    wandb.log({"roc_auc": roc_auc})
    
    predictions = predict(model, x_test)
    wandb.log({"classification_report": get_classification_report(predictions, y_test)})
    #wandb.log({"confussion matrix:": get_confussion_matrix(predictions, y_test)})

def get_score(model, x_test, y_test):
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test,y_pred)
    wandb.log({"learn_rate": score})
