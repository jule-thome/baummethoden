import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

'''
Read CSV
'''
#List with attribute names (it is optional to do this but it gives a better understanding of the data for a human reader)
attribute_names = ['variance_wavelet_transformed_image', 'skewness_wavelet_transformed_image', 'curtosis_wavelet_transformed_image', 'entropy_image', 'class']

#Read csv-file
data = pd.read_csv('data_banknote_authentication.csv', names=attribute_names)

#Shuffle data
data = data.sample(frac=1)

#Shows the first 5 rows of the data
print("Take a look at the data")
print(data.head())
print("\n")

'''
Understanding the Base Rate and Accuracy of a model
'''
print('Base rates are ...')
#Get the absolute number of how many instances in our data belong to class zero
count_real = len(data.loc[data['class']==0])
print('Real bills absolute: ' + str(count_real))

#Get the absolute number of how many instances in our data belong to class one
count_fake = len(data.loc[data['class']==1])
print('Fake bills abolute: ' +str(count_fake))

#Get the relative number of how many instances in our data belong to class zero
percentage_real = count_real/(count_fake+count_real)
print('Real bills in percent: ' + str(round(percentage_real,3)))

#Get the relative number of how many instances in our data belong to class one
percentage_fake = count_fake/(count_real+count_fake)
print('Fake bills in percent: ' + str(round(percentage_fake,3)))
print("\n")

'''
Splitting data into Training and Test Data
'''
#'class'-column
y_variable = data['class']

#all columns that are not the 'class'-column -> all columns that contain the attributes
x_variables = data.loc[:, data.columns != 'class']

#splits into training and test data
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=0.2)

# shapes of our data splits
print("Shapes of our training and test data split ...")
print(x_train.shape) 
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print("\n")

'''
Decision Tree Classifier
'''
print("--- Decision Tree Classifier ---")
#Create a classifier object 
classifier = DecisionTreeClassifier() 

#Classfier builds Decision Tree with training data
classifier = classifier.fit(x_train, y_train) 

#Shows importances of the attributes according to our model 
print("Importances of the attributes according to our model ...")
print(classifier.feature_importances_)
print("\n")

'''
Testing
'''
#Get predicted values from test data 
y_pred = classifier.predict(x_test)  

#Create the matrix that shows how often predicitons were done correctly and how often theey failed.
conf_mat = confusion_matrix(y_test, y_pred)

#The diagonal ones are the correctly predicted instances. The sum of this number devided by the number of all instances gives us the accuracy in percent.
accuracy = (conf_mat[0,0] + conf_mat[1,1]) /(conf_mat[0,0]+conf_mat[0,1]+ conf_mat[1,0]+conf_mat[1,1])

print('Accuracy: ' + str(round(accuracy,4)))
print('Confusion matrix:')
print(conf_mat)  
print('classification report:')
print(classification_report(y_test, y_pred))
print("\n")

'''
K-Folding
'''
#k_fold object
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

#scores reached with different splits of training/test data 
k_fold_scores = cross_val_score(classifier, x_variables, y_variable, cv=k_fold, n_jobs=1)

#arithmetic mean of accuracy scores 
mean_accuracy = np.mean(k_fold_scores)

print("K-Folding mean accuracy ...")
print(round(mean_accuracy, 4))
print("\n")

'''
GridSearch CV
'''
#tree parameters which shall be tested
tree_para = {'criterion':['gini','entropy'],'max_depth':[i for i in range(1,20)], 'min_samples_split':[i for i in range (2,20)]}

#GridSearchCV object
grd_clf = GridSearchCV(classifier, tree_para, cv=5)

#creates differnt trees with all the differnet parameters out of our data
grd_clf.fit(x_variables, y_variable)

#best paramters that were found
best_parameters = grd_clf.best_params_  
print(best_parameters)  

#new tree object with best parameters
model_with_best_tree_parameters = grd_clf.best_estimator_

#k_fold object
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

#scores reached with different splits of training/test data 
k_fold_scores = cross_val_score(model_with_best_tree_parameters, x_variables, y_variable, cv=k_fold, n_jobs=1)

#arithmetic mean of accuracy scores 
mean_accuracy_best_parameters_tree = np.mean(k_fold_scores)

print("K-Folding best tree ...")
print(round(mean_accuracy_best_parameters_tree, 4))
print("\n")

'''
Random Forest
'''
#RandomForestClassifier object
random_forest_classifier = RandomForestClassifier(n_estimators=10)

#list with accuracies with different test and training sets of Random Forest
accuracies_rand_forest = cross_val_score(random_forest_classifier, x_variables, y_variable, cv=k_fold, n_jobs=1)

##arithmetic mean of the list with the accuracies of the Random Forest
accuracy_rand = np.mean(accuracies_rand_forest)

print("--- Random Forest ---")
print('Accuracy Random Forest ' + str(round(accuracy_rand,4)))
print('Old accuracy: ' + str(round(mean_accuracy,4)))
print('Best tree accuracy: ' + str(round(mean_accuracy_best_parameters_tree,4)))