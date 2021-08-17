import pandas as pd
import pickle
from sklearn.metrics import classification_report,confusion_matrix

'''
Read test data and load baummethoden.pkl
'''
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

baummethoden = pickle.load(open( "baummethoden.pkl", "rb" ))


'''
Predictions
'''
prediction = baummethoden.predict(x_test)

# Show confusion matrix and classification report
print("Confusion matrix")
print(confusion_matrix(y_test,prediction))
print("\n")

print("Classification report")
print(classification_report(y_test,prediction))