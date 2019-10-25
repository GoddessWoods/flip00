import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from sklearn.preprocessing import label_binarize

if __name__ == '__main__':
    np.random.seed(0)
    data = pd.read_csv('train.csv')
    iris_types = data['type'].unique()
    n_class = iris_types.size
    x = data.iloc[:, :2]
    y = pd.Categorical(data['type']).codes
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=0)
    y_one_hot = label_binarize(y_test, np.arange(n_class))
    alpha = np.logspace(-2, 2, 20)
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