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