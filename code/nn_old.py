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


from sklearn.neural_network import MLPClassifier

nn_clf = MLPClassifier(max_iter=3000)

from sklearn.model_selection import GridSearchCV

grid_params = [{"hidden_layer_sizes":range(3,20), "activation":['identity', 'logistic', 'tanh', 'relu'], "solver":["lbfgs","sgd","adam"],"learning_rate":["adaptive"]}]
grid_search = GridSearchCV(nn_clf,param_grid=grid_params,cv=3,verbose=0)
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
print ('Best score of RandomForestClassifier : {}'.format(grid_search.best_score_))
print ('Best parameters of RandomForestClassifier : {}'.format(grid_search.best_params_))
print('RandomForestClassifier classification report')
print (metrics.classification_report(y_test, y_pred))
print ('\nAccuracy score of RandomForestClassifier is: '+str(metrics.accuracy_score(y_test, y_pred)))