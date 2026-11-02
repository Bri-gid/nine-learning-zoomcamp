import pandas as pd 
import numpy as np

car = pd.read_csv("car_fuel_efficiency_2026.csv")

print(car.head)
print(car) #question 2

print(car.columns)

print(car["fuel_type"].unique()) #question 3

print(car.columns.isnull().any().sum())#question 4

print(car[car['origin'] =="Asia"]['fuel_efficiency_mpg'].max()) #question 5

print(car['horsepower'].median())# question 6i

most_freq = car['horsepower'].mode()[0]

print(most_freq)# question 6ii

car['horsepower'] == car['horsepower'].fillna(most_freq) # question 6iii

print(car['horsepower'].median()) # question 6iv


asian_cars = car[car["origin"]=="Asia"]
X = asian_cars[["vehicle_weight","model_year"]].head(7).to_numpy()

XTX = X.T.dot(X)

XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv.dot(X.T).dot(y)
print(w.sum())
print(pd.__version__)

