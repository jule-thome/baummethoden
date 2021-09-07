import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeRegressor 
import pickle

#Read csv-file
data = pd.read_csv('data/auto-mpg.csv', sep=";")

#Shuffle data
data = data.sample(frac=1)

#'class'-column
y_variable = data['mpg']

#all columns that are not the 'class'-column -> all columns that contain the attributes
x_variables = data.loc[:, data.columns != 'mpg']


x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=0.2)

regressor = DecisionTreeRegressor() 

regressor = regressor.fit(x_train, y_train) 

y_pred = regressor.predict(x_test) 

file_to_write = open("models/baummethoden.pickle", "wb")
pickle.dump(regressor, file_to_write)