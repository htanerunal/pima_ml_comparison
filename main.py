# This file is created to compare ML Classifiers with NLC
# Copyright (c) 2021 Hamit Taner Ünal and Prof.Fatih Başçiftçi

# Classification Task for Pima Indian Diabetes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import std, median
from numpy import mean
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, make_scorer, accuracy_score, RocCurveDisplay, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, cross_validate
import warnings
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
warnings.filterwarnings('ignore')

#Determine seed
seed = 7
np.random.seed(seed)

# Read data
data = pd.read_csv("dataset.csv")

# Print data summary
print(data.head())
print(data.info())

# Split the columns to input and output
y = data.Outcome
x = data.drop('Outcome', axis = 1)

# Transform input data for better classification
scaler = StandardScaler()
X = scaler.fit_transform(x)

#Use seed to reproduce the results
print("Using seed ",seed)

# Define models to be used (with best hyperparameters)
model1 = LogisticRegression(C=1, max_iter=100, penalty='l2',solver='newton-cg')
model2 = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=8, bootstrap = True, max_features = 'log2')
model3 = Sequential()
model3.add(Dense(9, input_dim=X.shape[1], activation='relu'))
model3.add(Dense(3, activation='relu'))
model3.add(Dense(1, activation="sigmoid"))
model3.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
model4 = DecisionTreeClassifier(criterion='gini',max_depth=5,max_features='log2',min_samples_leaf=9,min_samples_split=3)
model5 = GaussianNB(var_smoothing=0.43287612810830584)
model6 = KNeighborsClassifier(leaf_size=1, metric='minkowski',n_neighbors=19, p = 1,weights='distance')
model7 = SVC(C=10,gamma=0.01,kernel='rbf')

#Define model3 (ANN) as Keras Classifier
# Prepare for ANN
def create_ANN_model(optimizer='RMSprop'):
    model = Sequential()
    model.add(Dense(9, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model3_keras = KerasClassifier(build_fn=create_ANN_model, verbose=0, epochs=50, batch_size=10)
model3_keras._estimator_type = "classifier"

# Define Stratified k-fold Cross Validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

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
    model3_keras.fit(X[train], y[train], verbose=0, epochs=150,batch_size=10)
    model4.fit(X[train], y[train])
    model5.fit(X[train], y[train])
    model6.fit(X[train], y[train])
    model7.fit(X[train], y[train])

    #Record scores
    score1 = model1.score(X[test], y[test])
    score2 = model2.score(X[test], y[test])
    score3 = model3_keras.score(X[test], y[test])
    score4 = model4.score(X[test], y[test])
    score5 = model5.score(X[test], y[test])
    score6 = model6.score(X[test], y[test])
    score7 = model7.score(X[test], y[test])

    #Get confusion matrices
    cf1 = confusion_matrix(y[test], model1.predict(X[test]))
    cf2 = confusion_matrix(y[test], model2.predict(X[test]))
    cf3 = confusion_matrix(y[test], model3_keras.predict(X[test]))
    cf4 = confusion_matrix(y[test], model4.predict(X[test]))
    cf5 = confusion_matrix(y[test], model5.predict(X[test]))
    cf6 = confusion_matrix(y[test], model6.predict(X[test]))
    cf7 = confusion_matrix(y[test], model7.predict(X[test]))



    # Print scores
    print("-----------------------------------------------------------------Fold:"+str(i))
    print("Accuracy 1.LR: %.2f%%" % (score1 * 100))
    print("Accuracy 2.RF: %.2f%%" % (score2 * 100))
    print("Accuracy 3.ANN: %.2f%%" % (score3 * 100))
    print("Accuracy 4.CART: %.2f%%" % (score4 * 100))
    print("Accuracy 5.NB: %.2f%%" % (score5 * 100))
    print("Accuracy 6.kNN: %.2f%%" % (score6 * 100))
    print("Accuracy 7.SVC: %.2f%%" % (score7 * 100))

    # Print confusion matrices
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
    print("Confusion Matrix for LR")
    print(cf1)
    print("Confusion Matrix for RF")
    print(cf2)
    print("Confusion Matrix for ANN")
    print(cf3)
    print("Confusion Matrix for CART")
    print(cf4)
    print("Confusion Matrix for NB")
    print(cf5)
    print("Confusion Matrix for KNN")
    print(cf6)
    print("Confusion Matrix for SVC")
    print(cf7)
    print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

    # Fill the scores array (to calculate mean and std later)
    overall_score1.append(score1 * 100)
    overall_score2.append(score2 * 100)
    overall_score3.append(score3 * 100)
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
print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

# Print Classification Reports
csv1 = []
csv2 = []
csv3 = []
csv4 = []
csv5 = []
csv6 = []
csv7 = []
print("Classification Report for Logistic Regression:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    cv_score = cross_val_score(model1, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv1.append(str(mean(cv_score)))
    csv1.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for Random Forest:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    cv_score = cross_val_score(model2, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv2.append(str(mean(cv_score)))
    csv2.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for ANN:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    cv_score = cross_val_score(model3_keras, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv3.append(str(mean(cv_score)))
    csv3.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for CART:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    cv_score = cross_val_score(model4, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv4.append(str(mean(cv_score)))
    csv4.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for Naive Bayes:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    cv_score = cross_val_score(model5, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv5.append(str(mean(cv_score)))
    csv5.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for kNN:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    cv_score = cross_val_score(model6, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv6.append(str(mean(cv_score)))
    csv6.append(str(std(cv_score)))
print("----------------------------------------------")

print("Classification Report for SVC:")
print("----------------------------------------------")
for score in [ "accuracy", "recall", "precision", "f1", "balanced_accuracy"]:
    cv_score = cross_val_score(model7, X, y, scoring=score, cv=kf)
    print(score + " Mean: %.8f" % mean(cv_score) + " STD: %.8f" % std(cv_score) + " Median: %.8f" % median(cv_score))
    csv7.append(str(mean(cv_score)))
    csv7.append(str(std(cv_score)))
print("----------------------------------------------")
print("***** Printing CSV Data to export *************")
print(csv1)
print(csv2)
print(csv3)
print(csv4)
print(csv5)
print(csv6)
print(csv7)
print("******** End of CSV ***************************")

# Plot ROC Curves
# Taken from scikit-learn user guide
# Special thanks to core developer and Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
#          and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
#          and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
#          Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.
# Citation: edregosa F et al. (2011) Scikit-learn: Machine learning in Python the Journal of machine Learning research 12:2825-2830

# Plot Logistic Regression

tprs1 = []
aucs1 = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(kf.split(X, y)):
    model1.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        model1,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs1.append(interp_tpr)

# Plot
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs1, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs1)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs1, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="ROC (LR)",
)
ax.legend(loc="lower right")
plt.show()

# Plot Random Forest

tprs2 = []
aucs2 = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(kf.split(X, y)):
    model2.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        model2,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs2.append(interp_tpr)

# Plot
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs2, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs2)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs2, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="ROC (RF)",
)
ax.legend(loc="lower right")
plt.show()

# Plot ANN

tprs3 = []
aucs3 = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(kf.split(X, y)):
    model3_keras.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        model3_keras,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs3.append(interp_tpr)

# Plot
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs3, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs3)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs3, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="ROC (Keras-ANN)",
)
ax.legend(loc="lower right")
plt.show()

# Plot CART

tprs4 = []
aucs4 = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(kf.split(X, y)):
    model4.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        model4,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs4.append(interp_tpr)

# Plot
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs4, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs4)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs4, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="ROC (CART)",
)
ax.legend(loc="lower right")
plt.show()

# Plot NB

tprs5 = []
aucs5 = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(kf.split(X, y)):
    model5.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        model5,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs5.append(interp_tpr)

# Plot
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs5, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs5)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs5, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="ROC (NB)",
)
ax.legend(loc="lower right")
plt.show()

# Plot KNN

tprs6 = []
aucs6 = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(kf.split(X, y)):
    model6.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        model6,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs6.append(interp_tpr)

# Plot
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs6, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs6)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs6, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="ROC (KNN)",
)
ax.legend(loc="lower right")
plt.show()

# Plot SVC

tprs7 = []
aucs7 = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(kf.split(X, y)):
    model7.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        model7,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs7.append(interp_tpr)

# Plot
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs7, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs7)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs7, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="ROC (CART)",
)
ax.legend(loc="lower right")
plt.show()




