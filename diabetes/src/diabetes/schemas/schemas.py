from pydantic import BaseModel

class Person(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI:float
    DiabetesPedigreeFunction: float
    Age:float 
    Outcome: float

class ModelEval(BaseModel):
    accuracy:float
    roc_auc:float