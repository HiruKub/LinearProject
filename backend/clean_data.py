import numpy as np
import pandas as pd

csv_path = "data/diabetes.csv"
csv_file = pd.read_csv(csv_path)

mean_field = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age","Pregnancies","DiabetesPedigreeFunction"]
mean_values = (csv_file[mean_field]==(csv_file[mean_field]).mean()).sum()
print(mean_values)

sd_field = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age","Pregnancies","DiabetesPedigreeFunction"]
sd_values = (csv_file[sd_field]==(csv_file[sd_field]).median()).sum()
print("sd_values")
print(sd_values)

zero_field = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age","Pregnancies","DiabetesPedigreeFunction"]
missing_values = (csv_file[zero_field] == 0).sum()
print(missing_values)