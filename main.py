# This file is created to compare ML Classifiers with NLC
# Copyright (c) 2021 Hamit Taner Ünal and Prof.Fatih Başçiftçi

# Classification Task for Pima Indian Diabetes

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from numpy import mean
from numpy import std
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Read data
data = pd.read_csv("dataset.csv")

# Print data summary
print(data.head())
print(data.info())

# Split the columns to input and output
y = data.Outcome
x = data.drop('Outcome', axis = 1)
columns = x.columns

# Transform input data for better classification
scaler = StandardScaler()
X = scaler.fit_transform(x)

# Define models to be used (with hyperparameters)
model1 = LogisticRegression(C=1, max_iter=100, penalty='l2',solver='newton-cg')
model2 = RandomForestClassifier(n_estimators=400, criterion='entropy', max_depth=8, bootstrap = True, max_features = 'log2')
model3 = Sequential()
model3.add(Dense(9, input_dim=8, activation='relu'))
model3.add(Dense(3, activation='relu'))
model3.add(Dense(1, activation="sigmoid"))
model4 = DecisionTreeClassifier(criterion='entropy',max_depth=5,max_features='log2',min_samples_leaf=9,min_samples_split=5)
model5 = GaussianNB(var_smoothing=0.43287612810830584)
model6 = KNeighborsClassifier(leaf_size=1, metric='minkowski',n_neighbors=19, p = 1,weights='distance')
model7 = SVC(C=10,gamma=0.01,kernel='rbf')

# Define Stratified k-fold Cross Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)

# Initialize scores for each classifier
overall_score1 = []
overall_score2 = []
overall_score3 = []
overall_score4 = []
overall_score5 = []
overall_score6 = []
overall_score7 = []

# Start k-fold cross validation loop
# This splits the data in every fold as training and test sets
i=0
for train, test in kf.split(X, y):
    i+=1

    # Fit models
    model1.fit(X[train], y[train])
    model2.fit(X[train], y[train])
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model3.fit(X[train], y[train], verbose=0, epochs=150, batch_size=10)
    model4.fit(X[train], y[train])
    model5.fit(X[train], y[train])
    model6.fit(X[train], y[train])
    model7.fit(X[train], y[train])

    #Record scores
    score1 = model1.score(X[test], y[test])
    score2 = model2.score(X[test], y[test])
    score3 = model3.evaluate(X[test], y[test], verbose=0)
    score4 = model4.score(X[test], y[test])
    score5 = model5.score(X[test], y[test])
    score6 = model6.score(X[test], y[test])
    score7 = model7.score(X[test], y[test])

    # Print scores
    print("------------------------Fold:"+str(i))
    print("Accuracy 1.LR: %.2f%%" % (score1 * 100))
    print("Accuracy 2.RF: %.2f%%" % (score2 * 100))
    print("Accuracy 3.ANN: %.2f%%" % (score3[1] * 100))
    print("Accuracy 4.CART: %.2f%%" % (score4 * 100))
    print("Accuracy 5.NB: %.2f%%" % (score5 * 100))
    print("Accuracy 6.kNN: %.2f%%" % (score6 * 100))
    print("Accuracy 7.SVC: %.2f%%" % (score7 * 100))

    # Fill the scores array (to calculate mean and std later)
    overall_score1.append(score1 * 100)
    overall_score2.append(score2 * 100)
    overall_score3.append(score3[1] * 100)
    overall_score4.append(score4 * 100)
    overall_score5.append(score5 * 100)
    overall_score6.append(score6 * 100)
    overall_score7.append(score7 * 100)

# Now print overall scores
print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
print("OVERALL SCORES")
print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
print("Result 1.LR: %.2f%% (+/- %.2f%%)" % (np.mean(overall_score1), np.std(overall_score1)))
print("Result 2.RF: %.2f%% (+/- %.2f%%)" % (np.mean(overall_score2), np.std(overall_score2)))
print("Result 3.ANN: %.2f%% (+/- %.2f%%)" % (np.mean(overall_score3), np.std(overall_score3)))
print("Result 4.CART: %.2f%% (+/- %.2f%%)" % (np.mean(overall_score4), np.std(overall_score4)))
print("Result 5.NB: %.2f%% (+/- %.2f%%)" % (np.mean(overall_score5), np.std(overall_score5)))
print("Result 6.kNN: %.2f%% (+/- %.2f%%)" % (np.mean(overall_score6), np.std(overall_score6)))
print("Result 7.SVC: %.2f%% (+/- %.2f%%)" % (np.mean(overall_score7), np.std(overall_score7)))