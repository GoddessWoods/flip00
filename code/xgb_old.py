import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
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

model = xgb.XGBClassifier(nthread=4,
                          learning_rate=0.08,
                          n_estimators=50,
                          max_depth=5,
                          gamma=0,
                          subsample=0.9,
                          colsample_bytree=0.5)


model.fit(X_train.values, y_train.values)
y_pred_test = model.predict_proba(X_test.values)
print(y_pred_test)
#xgb_test_auc = roc_auc_score(pd.get_dummies(y_test), y_pred_test)
#print('xgboost test auc: %.5f' % xgb_test_auc)

y_pred = model.predict(X_test.values)
print (metrics.classification_report(y_test.values, y_pred))
print ('\nAccuracy score is: '+str(metrics.accuracy_score(y_test.values, y_pred)))
