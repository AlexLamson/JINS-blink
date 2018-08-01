'''
To do (sorted approximately by priority)
========================================
S -> get rid of very weird features so it becomes easier to interpret
A -> double check f1 scores
S -> email f1 scores & accuracy scores & stuff
S -> email graph of all feature importances
S -> email graph of top feature importances
S -> check that feature importances are computed the same as in "EarBit: Using Wearable Sensors to Detect Eating Episodes in Unconstrained Environments"
SA-> make one file for each plot (move the code that compute X_all & y_all to a file too)
A -> add legends to clusterings plots
S -> add ensemble model


General Notes
=============

768 features

Leave One Subject Out
    mean of accuracy:
    std of accuracy:
    mean f1 score:

'''
import time
import datetime
import pickle
import pydotplus
from tqdm import tqdm
import itertools
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GMM
from sklearn.cluster import SpectralClustering, KMeans
from math import floor

from util import save_object, load_object
from prepare_data import get_data



X_all, y_all, groups, feature_names, subjects, labels, class_names = get_data(use_precomputed=False)




print("testing model")
test_accuracies = []
train_accuracies = []
f1_scores = []
cnf_matrix_sum = None

logo = LeaveOneGroupOut()
for train_index, test_index in tqdm(logo.split(X_all, y_all, groups), total=len(set(groups))):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]
    # print(X_train, X_test, y_train, y_test)

    # model = DecisionTreeClassifier(max_depth=8, min_samples_split=50, min_samples_leaf=40, min_impurity_split=0.15)

    # weights = {1:1, 2:1, 3:1, 4:3, 5:1} #, class_weight=weights
    # model = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_split=50, min_samples_leaf=40)
    model = RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_split=30, min_samples_leaf=30)

    # model = KNeighborsClassifier(n_neighbors=n_neighbors)
    # model = LinearSVC()
    # model = SVC()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_accuracy = model.score(X_test, y_test)
    train_accuracy = model.score(X_train, y_train)
    test_accuracies += [test_accuracy]
    train_accuracies += [train_accuracy]

    my_f1_score = f1_score(y_test, y_pred, average=None)
    # print(my_f1_score)
    f1_scores += [my_f1_score]


    cnf_matrix = confusion_matrix(y_test, model.predict(X_test))
    if cnf_matrix_sum is None:
        cnf_matrix_sum = cnf_matrix
    else:
        cnf_matrix_sum += cnf_matrix


    # # Visualize the feature importance
    # importance = model.feature_importances_

    # # zipped = list(reversed(sorted(zip(importance, feature_names), reverse=True)))
    # zipped = list(reversed(sorted(zip(importance, feature_names), reverse=True)[:30]))
    # zipped = [x for x in zipped if x[0] > 0]
    # importance, feature_names = [x[0] for x in zipped], [x[1] for x in zipped]

    # importance = np.array(importance)
    # # exit()

    # plt.barh(np.arange(importance.size), importance)
    # if len(zipped) < 60:
    #     plt.yticks(np.arange(importance.size), feature_names)
    # # plt.xscale('log')
    # plt.tight_layout()
    # plt.show()
    # exit()


test_accuracies = np.array(test_accuracies)
train_accuracies = np.array(train_accuracies)
f1_scores = np.array(f1_scores)

print("mean train accuracy: {}".format(np.mean(train_accuracy)))
print("mean test accuracy: {}".format(np.mean(test_accuracy)))
print("std dev test accuracy: {:.3f}".format(np.std(test_accuracies)))
print("mean f1 scores: {}".format(np.mean(f1_scores, axis=0)))



# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names[:5], normalize=True, title='Normalized confusion matrix')
plt.show()



