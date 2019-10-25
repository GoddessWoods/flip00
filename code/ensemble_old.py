import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics  import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, cross_val_score

data = pd.read_csv('train.csv')
traindf = data.drop(['id'], axis = 1)
df = pd.get_dummies(traindf.drop('type', axis = 1))
X_train, X_test, y_train, y_test = train_test_split(df, traindf['type'], test_size = 0.2, random_state = 42)
X_train = pd.get_dummies(traindf.drop('type', axis = 1))
y_train = traindf['type']
accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)

# Using GridSearchCV for the RandomForestClassifier with cv = 10
params = {'n_estimators' : [10, 20, 50, 100], 'criterion' : ['gini', 'entropy'],
          'max_depth' : [None, 1, 2, 3, 5], 'max_features' : ['auto', 'sqrt', 'log2', None]}
rf = RandomForestClassifier(random_state = 0)
clf = GridSearchCV(rf, param_grid = params, scoring = accuracy_scorer, cv = 10, n_jobs = -1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Best score of RandomForestClassifier : {}'.format(clf.best_score_))
print ('Best parameters of RandomForestClassifier : {}'.format(clf.best_params_))
print('RandomForestClassifier classification report')
print (metrics.classification_report(y_test, y_pred))
print ('\nAccuracy score of RandomForestClassifier is: '+str(metrics.accuracy_score(y_test, y_pred)))

# Using GridSearchCV for the BaggingClassifier with cv = 10
params = {'n_estimators' : [10, 20, 50, 100], 'max_samples' : [1, 3, 5, 10]}
bag = BaggingClassifier(random_state = 0)
clf = GridSearchCV(bag, param_grid = params, scoring = accuracy_scorer, cv = 10, n_jobs = -1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Best score of BaggingClassifier: {}'.format(clf.best_score_))
print ('Best parameters of BaggingClassifier: {}'.format(clf.best_params_))
#bag_best = BaggingClassifier(max_samples = 5, n_estimators = 20, random_state = 0)
print('BaggingClassifier classification report')
print (metrics.classification_report(y_test, y_pred))
print ('\nAccuracy score of BaggingClassifier is: '+str(metrics.accuracy_score(y_test, y_pred)))

# Using GridSearchCV for the GradientBoostingClassifier with cv = 5
params = {'learning_rate' : [0.05, 0.07, 0.1, 0.3, 0.5], 'n_estimators' : [10, 20, 50, 100, 200], 'max_depth' : [1, 2, 3, 5]}
gbc = GradientBoostingClassifier(random_state = 0)
clf = GridSearchCV(gbc, param_grid = params, scoring = accuracy_scorer, cv = 10, n_jobs = -1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Best score of GradientBoostingClassifier: {}'.format(clf.best_score_))
print ('Best parameters of GradientBoostingClassifier: {}'.format(clf.best_params_))
print('GradientBoostingClassifier calssification report')
print (metrics.classification_report(y_test, y_pred))
print ('\nAccuracy score of GradientBoostingClassifier is: '+str(metrics.accuracy_score(y_test, y_pred)))
print(confusion_matrix(y_test, y_pred))

# Using GridSearchCV for the KNN classifier with CV = 5
params = {'n_neighbors' : [3, 5, 10, 20], 'leaf_size' : [10, 20, 30, 50], 'p' : [1, 2, 5],
          'weights' : ['uniform', 'distance'], 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']}
knc = KNeighborsClassifier()
clf = GridSearchCV(knc, param_grid = params, scoring = accuracy_scorer, cv = 10, n_jobs = -1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Best score of KNN: {}'.format(clf.best_score_))
print ('Best parameters of KNN: {}'.format(clf.best_params_))
print('KNN classification report')
print (metrics.classification_report(y_test, y_pred))
print ('\nAccuracy score of KNN is: '+str(metrics.accuracy_score(y_test, y_pred)))

# Using GridSearchCV for LogisticRegression with CV = 10
params = {'penalty' : ['l1', 'l2'], 'C' : [1, 2, 3, 5, 10]}
lr = LogisticRegression(random_state = 0)
clf = GridSearchCV(lr, param_grid = params, scoring = accuracy_scorer, cv = 10, n_jobs = -1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Best Score of LogisticRegression: {}'.format(clf.best_score_))
print ('Best parameters of LogisticRegression: {}'.format(clf.best_params_))
print('LogisticRegression classification report')
print (metrics.classification_report(y_test, y_pred))
print ('\nAccuracy score of LogisticRegression is: '+str(metrics.accuracy_score(y_test, y_pred)))

# Using GridSearchCV for SVC with CV = 10
params = {'kernel' : ['linear', 'rbf'], 'C' : [1, 3, 5, 10], 'degree' : [3, 5, 10]}
svc = SVC(probability = True, random_state = 0)
clf = GridSearchCV(svc, param_grid = params, scoring = accuracy_scorer, cv = 10, n_jobs = -1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ('Best score of SVC: {}'.format(clf.best_score_))
print ('Best parameters of SVC: {}'.format(clf.best_params_))
print('SVC classification report')
print (metrics.classification_report(y_test, y_pred))
print ('\nAccuracy score of SVC is: '+str(metrics.accuracy_score(y_test, y_pred)))

rf_best = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = 5, random_state = 0,max_features=None)
bag_best = BaggingClassifier(max_samples = 10, n_estimators = 100, random_state = 0)
gbc_best = GradientBoostingClassifier(n_estimators = 20, max_depth = 2, learning_rate = 0.3, random_state = 0)
lr_best = LogisticRegression(C = 1, penalty = 'l1')
svc_best = SVC(C = 10, degree = 3, kernel = 'linear')
knc_best = KNeighborsClassifier(algorithm='auto', leaf_size=10, n_neighbors= 20, p=5, weights='uniform')

# voting
voting_clf = VotingClassifier(estimators = [('rf', rf_best), ('bag', bag_best), ('gbc', gbc_best), ('lr', lr_best), ('svc', svc_best), ('knc',knc_best)],voting = 'hard')
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)
y_pred = voting_clf.predict(X_test)
print(y_pred)
print ('\nAccuracy score for VotingClassifier is : '+str(voting_clf.score(X_train, y_train)))
print (metrics.classification_report(y_test, y_pred))
print ('\nAccuracy score is: '+str(metrics.accuracy_score(y_test, y_pred)))