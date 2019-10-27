import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.preprocessing import label_binarize
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


if __name__ == '__main__':
    data = pd.read_csv('train.csv')
    #traindf = data.drop(['id'], axis=1)
    #df = pd.get_dummies(traindf.drop('type', axis=1))
    #X_train, X_test, y_train, y_test = train_test_split(df, traindf['type'], test_size=0.2, random_state=42)
    #X_train = pd.get_dummies(traindf.drop('type', axis=1))
    #y_train = traindf['type']
    accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)



    np.random.seed(0)
    data = pd.read_csv('train.csv')
    iris_types = data['type'].unique()
    n_class = iris_types.size
    x = data.iloc[:, :2]
    y = pd.Categorical(data['type']).codes
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    y_one_hot = label_binarize(y_test, np.arange(n_class))
    alpha = np.logspace(-2, 2, 20)

    rf_best = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5, random_state=0,
                                     max_features=None)
    bag_best = BaggingClassifier(max_samples=10, n_estimators=100, random_state=0)
    gbc_best = GradientBoostingClassifier(n_estimators=20, max_depth=2, learning_rate=0.3, random_state=0)
    lr_best = LogisticRegression(C=1, penalty='l1')
    svc_best = SVC(C=10, degree=3, kernel='linear')
    knc_best = KNeighborsClassifier(algorithm='auto', leaf_size=10, n_neighbors=20, p=5, weights='uniform')

    # voting
    '''
    model = VotingClassifier(
        estimators=[('rf', rf_best), ('bag', bag_best), ('gbc', gbc_best), ('lr', lr_best), ('svc', svc_best),
                    ('knc', knc_best)],voting='soft')
    #voting_clf.fit(x_train, y_train)
    #y_pred = voting_clf.predict(x_test)
    #y_pred = voting_clf.predict(x_test)
    '''

    model = LogisticRegressionCV(Cs=alpha, cv=3, penalty='l2')
    model.fit(x_train, y_train)
    print( model.C_)

    y_score = model.predict_proba(x_test)

    print( metrics.roc_auc_score(y_one_hot, y_score, average='micro'))

    fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
    auc = metrics.auc(fpr, tpr)
    print(auc)

    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False

    plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label=u'AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title('ROC and AUC', fontsize=17)
    plt.show()