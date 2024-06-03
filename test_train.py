from orchestration.train_templates import TrainModels
import pandas as pd

templates = [
                {"model" : "tree",  "init" : {"scale_input" : False},  "fit" :{"cv" :5}},
                {"model" : "knn",  "init" : {"scale_input" : False},  "fit" :{"cv" :5}},
                {"model" : "xgboost",  "init" : {"scale_input" : False},  "fit" :{"cv" :5}},
                {"model" : "lightgbm",  "init" : {"scale_input" : False},  "fit" :{"cv" :5}},
            ]

df = pd.read_csv("data/sample.csv")
features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
target = "Outcome"

train_obj = TrainModels(templates)
mdl_objs = train_obj.train_models(df, features = features, target = target, cv=5, test_size=0.2, random_state=42)