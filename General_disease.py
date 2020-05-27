import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset=pd.read_csv('data.csv')
dataset.drop(dataset[dataset['Disease']==0].index, inplace = True)
X = dataset.iloc[:,1:].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40, random_state = 0)


# Fitting RandomForest classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 22, criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)


pickle.dump(classifier,open('model.pkl', 'wb'))