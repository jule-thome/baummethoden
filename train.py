import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


'''
Read CSV
'''
#List with attribute names (it is optional to do this but it gives a better understanding of the data for a human reader)
attribute_names = ['variance_wavelet_transformed_image', 'skewness_wavelet_transformed_image', 'curtosis_wavelet_transformed_image', 'entropy_image', 'class']

#Read csv-file
data = pd.read_csv('data/data_banknote_authentication.csv', names=attribute_names)

#Shuffle data
data = data.sample(frac=1)


'''
Splitting data into Training and Test Data
'''
#'class'-column
y_variable = data['class']

#all columns that are not the 'class'-column -> all columns that contain the attributes
x_variables = data.loc[:, data.columns != 'class']

#splits into training and test data
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=0.2)

#save test data for later predictions
x_test.to_csv("data/x_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)


'''
Random Forest
'''
#RandomForestClassifier object
random_forest_classifier = RandomForestClassifier(n_estimators=10)

#Train random forest
random_forest_classifier.fit(x_train, y_train)
print("Model trained successfully.")


'''
Model save using pickle
'''
pickle.dump(random_forest_classifier, open("models/baummethoden.pkl", "wb"))
print("Model pickled successfully.")