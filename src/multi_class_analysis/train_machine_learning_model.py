from tqdm import tqdm
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from util import save_object, load_object
from prepare_data import get_data


use_precomputed = False
X_all, y_all, groups, feature_names, subjects, labels, class_names, is_moving_data = get_data(use_precomputed=use_precomputed)


# def f1_score(y_test, y_pred, average=None):
#     pres = []
#     labels = sorted(set(y_test))
#     for label in labels:
#         tp = np.sum((y_pred == label) & (y_test == label))
#         fp = np.sum((y_pred == label) & (y_test != label))
#         # print("tp:{} fp:{}".format(tp, fp))
#         pre = tp/(tp+fp)
#         pres += [pre]
#     pre_macro = sum(pres)/len(pres)

#     f1s = []
#     for label in labels:
#         # print("checking label {}".format(label))
#         tp = np.sum((y_pred == label) & (y_test == label))
#         fp = np.sum((y_pred == label) & (y_test != label))
#         fn = np.sum((y_pred != label) & (y_test == label))
#         pre = pre_macro

#         rec = tp/(fn+tp)

#         f1 = 2 * (pre*rec) / (pre+rec)
#         import math
#         if math.isnan(f1):
#             # print("tp: {} fp: {} fn: {}".format(tp, fp, fn))
#             # print("never predicted label {}".format(label))
#             print("y_test: {}".format(y_test))
#             print("y_pred: {}".format(y_pred))
#             # print("foo: {}".format((y_test == y_pred).astype(int)))
#             f1 = 0
#         f1s += [f1]


#     # print(f1s)
#     return f1s





print("training & testing model using Leave One Out")
test_accuracies = []
train_accuracies = []
f1_scores = []
cnf_matrix_sum = None

logo = LeaveOneGroupOut()
i = 0
for train_index, test_index in tqdm(logo.split(X_all, y_all, groups), total=len(set(groups))):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]
    # print(X_train, X_test, y_train, y_test)

    # model = DecisionTreeClassifier(max_depth=8, min_samples_split=50, min_samples_leaf=40, min_impurity_split=0.15)

    # weights = {1:1, 2:1, 3:1, 4:3, 5:1} #, class_weight=weights
    # model = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_split=50, min_samples_leaf=40)
    model = RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_split=30, min_samples_leaf=30)
    # model = GradientBoostingClassifier(n_estimators=20, max_depth=3, min_samples_split=30, min_samples_leaf=30)

    # model = KNeighborsClassifier(n_neighbors=3)
    # model = LinearSVC()
    # model = SVC()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    train_accuracies += [train_accuracy]
    test_accuracies += [test_accuracy]

    i = i + 1
    labels = sorted(set(y_test))
    for label in labels:
        accuracy = np.sum((y_pred == label) & (y_test == label)) / np.sum(y_test == label)
        # print("subject: {} label {} accuracy: {}".format(subjects[i], label, accuracy))

    my_f1_score = f1_score(y_test, y_pred, average=None)
    # print(my_f1_score)
    f1_scores += [my_f1_score]


    cnf_matrix = confusion_matrix(y_test, model.predict(X_test))
    if cnf_matrix_sum is None:
        cnf_matrix_sum = cnf_matrix
    else:
        cnf_matrix_sum = cnf_matrix_sum + cnf_matrix


train_accuracies = np.array(train_accuracies)
test_accuracies = np.array(test_accuracies)
f1_scores = np.array(f1_scores)

print("mean train accuracy: {}".format(np.mean(train_accuracies)))
print("mean test accuracy: {}".format(np.mean(test_accuracies)))

print("min test accuracy: {}".format(np.min(test_accuracies)))
print("max test accuracy: {}".format(np.max(test_accuracies)))

print("std dev test accuracy: {:.3f}".format(np.std(test_accuracies)))
print("mean f1 score for each class: {}".format(np.mean(f1_scores, axis=0)))
print("mean f1 score: {:.5f}".format(np.mean(np.mean(f1_scores, axis=0))))


# save the f1-scores
print("saving f1 scores")
f1_score_filename = "f1_scores_stationary.pkl" if is_moving_data else "f1_scores_mobile.pkl"
save_object(f1_score_filename, f1_scores)

# Visualize the feature importance
importance = model.feature_importances_

# zipped = list(reversed(sorted(zip(importance, feature_names), reverse=True))) #all features
# zipped = list(reversed(sorted(zip(importance, feature_names), reverse=True)))[-30:] #worst features
zipped = list(reversed(sorted(zip(importance, feature_names), reverse=True)[:30])) #best features
zipped = [x for x in zipped if x[0] > 0]
importance, feature_names = [x[0] for x in zipped], [x[1] for x in zipped]
importance = np.array(importance)

plt.barh(np.arange(importance.size), importance)
if len(zipped) < 60:
    plt.yticks(np.arange(importance.size), feature_names)
# plt.xscale('log')
plt.tight_layout()
plt.show()
# exit()



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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
plot_confusion_matrix(cnf_matrix_sum, classes=class_names[:5], normalize=True, title='Normalized confusion matrix')
plt.show()
