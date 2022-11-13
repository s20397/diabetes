"""
This is a boilerplate pipeline 'data_processing_diabetes'
generated using Kedro 0.18.3
"""
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

def preprocess_diabetes(diabetes: pd.DataFrame):
    diabetes['Glucose'] = diabetes['Glucose'].replace(0,diabetes['Glucose'].mean())
    diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(0,diabetes['BloodPressure'].mean())
    diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(0,diabetes['SkinThickness'].mean())
    diabetes['Insulin'] = diabetes['Insulin'].replace(0,diabetes['Insulin'].mean())
    diabetes['BMI'] = diabetes['BMI'].replace(0,diabetes['BMI'].mean())
    return diabetes

def splitData(data: pd.DataFrame):
    x = data.drop('Outcome',axis=1)
    y = data['Outcome']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=101)
    return x_train,x_test,y_train,y_test

def createModel():
    model = LogisticRegression()
    return model

def normalizeFeatures(data):
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data

def trainModel(model, x_train, y_train):
    model.fit(x_train,y_train)
    return model

def predict(model, x_test):
    predictions = model.predict(x_test)
    return predictions

def getClassificationReport(predictions, y_test):
    report = classification_report(y_test,predictions)
    return report

def getConfussionMatrix(predictions, y_test):
    matrix = confusion_matrix(y_test,predictions)
    return matrix

def evaluateModel(model, x_test, y_test):
    predictions = predict(model, x_test)
    print(getClassificationReport(predictions, y_test))
    print("confussion matrix:")
    print(getConfussionMatrix(predictions, y_test))
