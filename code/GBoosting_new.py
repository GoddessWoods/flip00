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
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

data = pd.read_csv('train.csv')
data['hair_soul'] = data['hair_length'] * data['has_soul']
data['hair_bone'] = data['hair_length'] * data['bone_length']
#data['bone_soul'] = data.apply(lambda row: row['bone_length']*row['has_soul'],axis=1)
data['hair_soul_bone'] = data['hair_length'] * data['has_soul'] * data['bone_length']

traindf = data.drop(['id',"bone_length", "color"], axis = 1)
df = pd.get_dummies(traindf.drop('type', axis = 1))
X_train, X_test, y_train, y_test = train_test_split(df, traindf['type'], test_size = 0.2, random_state = 42)
X_train = pd.get_dummies(traindf.drop('type', axis = 1))
y_train = traindf['type']
accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)

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