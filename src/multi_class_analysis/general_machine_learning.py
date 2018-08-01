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
# from feature_extractor import get_features
from prepare_data import get_data



X_all, y_all, groups, feature_names, subjects, labels, class_names = get_data(use_precomputed=False)



# print("loading pickled data")
# (X_all, y_all, groups, feature_names) = load_object("all_data.pkl")


# # accumulate the data for the all the subjects
# print("loading raw data into memory")
# for subject in subjects:
#     # subject_data = np.zeros(shape=(0,201,10))
#     for label in labels:
#         path = "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/jins/{0}_label{1}.mat".format(subject, label)

#         # [ trial * window frames * sensor channels ]
#         matlab_object = scipy.io.loadmat(path)

#         subject_matrix = matlab_object['data_chunk']
#         # subject_data = np.concatenate((subject_data, subject_matrix), axis=0)
#         groups += [subject]*subject_matrix.shape[0]
#         y_all += [label]*subject_matrix.shape[0]

#         for trial in range(subject_matrix.shape[0]):
#             raw_window = subject_matrix[trial,:,:]
#             # print(raw_window.shape)
#             if X_all_raw is None:
#                 X_all_raw = np.empty(shape=(0,len(raw_window), 10), dtype=float)
#             # print(X_all_raw.shape)
#             # exit()
#             X_all_raw = np.concatenate((X_all_raw, raw_window[np.newaxis,:,:]), axis=0)


# print("normalizing data")
# # normalize accelerometer signals
# a = np.mean(np.std(X_all_raw[:,:,0:3], axis=2))
# X_all_raw[:,:,0:3] = X_all_raw[:,:,0:3] / a

# # normalize gyroscope signals
# a = np.mean(np.std(X_all_raw[:,:,3:6], axis=2))
# X_all_raw[:,:,3:6] = X_all_raw[:,:,3:6] / a

# # normalize eog signals
# a = np.mean(np.std(X_all_raw[:,:,6:], axis=2))
# X_all_raw[:,:,6:10] = X_all_raw[:,:,6:10] / a


# print("extracting features")
# for trial in tqdm(range(X_all_raw.shape[0])):
#     feature_extracted_window, feature_names = get_features(X_all_raw[trial,:,:])
#     feature_extracted_window = np.array(feature_extracted_window)

#     if X_all is None:
#         X_all = np.empty(shape=(0,len(feature_extracted_window)), dtype=float)
#     X_all = np.concatenate((X_all, feature_extracted_window[np.newaxis,:]), axis=0)


# y_all = np.array(y_all)
# # np.savetxt("y_all.txt", y_all)  # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
# groups = np.array(groups)

# print("pickling data")
# save_object("all_data.pkl", (X_all, y_all, groups, feature_names))




# print("reducing dimensions with PCA")
# plt.figure(1)
# pca = PCA(n_components=2)
# pca.fit(X_all)
# X_all_reduced = pca.transform(X_all)
# # print(X_all_reduced.shape)
# plt.scatter(X_all_reduced[:,0], X_all_reduced[:,1], c=y_all)
# plt.set_cmap('hsv')
# plt.title("PCA n=2")
# # plt.show()



# plt.figure(2)
# print("reducing dimensions with t-SNE")
# tsne = TSNE(n_components=2)
# X_all_reduced = tsne.fit_transform(X_all)
# # print(X_all_reduced.shape)
# plt.scatter(X_all_reduced[:,0], X_all_reduced[:,1], c=y_all)
# plt.set_cmap('hsv')
# plt.title("t-SNE n=2")
# # plt.show()



# print("reducing dimensions with GMM then PCA")
# plt.figure(3)
# gmm = GMM(n_components=6, covariance_type='tied')
# gmm.fit(X_all)
# X_all_reduced = gmm.predict_proba(X_all)

# pca = PCA(n_components=2)
# pca.fit(X_all_reduced)
# X_all_reduced = pca.transform(X_all_reduced)

# def rand_jitter(arr):
#     stdev = .05*(max(arr)-min(arr))
#     return arr + np.random.randn(len(arr)) * stdev

# X_all_reduced[:,0] = rand_jitter(X_all_reduced[:,0])
# X_all_reduced[:,1] = rand_jitter(X_all_reduced[:,1])

# # print(X_all_reduced.shape)
# plt.scatter(X_all_reduced[:,0], X_all_reduced[:,1], c=y_all)
# plt.set_cmap('hsv')
# plt.title("data -> GMM n={} -> PCA n=2".format(n_components))
# # plt.show()



# n_components = 7
# print("reducing dimensions with spectral clustering then PCA")
# plt.figure(4)
# spectral = SpectralClustering(n_clusters=n_components)
# # spectral.fit(X_all)
# X_all_reduced = spectral.fit_predict(X_all)

# # one-hot-ify it
# b = np.zeros((X_all_reduced.shape[0], n_components))
# b[np.arange(X_all_reduced.shape[0]), X_all_reduced] = 1
# X_all_reduced = b

# pca = PCA(n_components=2)
# pca.fit(X_all_reduced)
# X_all_reduced = pca.transform(X_all_reduced)


# def rand_jitter(arr):
#     stdev = .05*(max(arr)-min(arr))
#     return arr + np.random.randn(len(arr)) * stdev


# X_all_reduced[:,0] = rand_jitter(X_all_reduced[:,0])
# X_all_reduced[:,1] = rand_jitter(X_all_reduced[:,1])

# # print(X_all_reduced.shape)
# plt.scatter(X_all_reduced[:,0], X_all_reduced[:,1], c=y_all)
# plt.set_cmap('hsv')
# plt.title("data -> spectral clustering n={} -> PCA n=2".format(n_components))
# # plt.show()



# n_components = 5
# print("reducing dimensions with k-means clustering then PCA")
# plt.figure(5)
# kmeans = KMeans(n_clusters=n_components)
# # kmeans.fit(X_all)
# X_all_reduced = kmeans.fit_predict(X_all)

# # one-hot-ify it
# b = np.zeros((X_all_reduced.shape[0], n_components))
# b[np.arange(X_all_reduced.shape[0]), X_all_reduced] = 1
# X_all_reduced = b

# pca = PCA(n_components=2)
# pca.fit(X_all_reduced)
# X_all_reduced = pca.transform(X_all_reduced)

# def rand_jitter(arr):
#     stdev = .05*(max(arr)-min(arr))
#     return arr + np.random.randn(len(arr)) * stdev

# X_all_reduced[:,0] = rand_jitter(X_all_reduced[:,0])
# X_all_reduced[:,1] = rand_jitter(X_all_reduced[:,1])

# # print(X_all_reduced.shape)
# plt.scatter(X_all_reduced[:,0], X_all_reduced[:,1], c=y_all)
# plt.set_cmap('hsv')
# plt.title("data -> k-means clustering n={} -> PCA n=2".format(n_components))
# plt.show()



# plt.show()
# exit()


if True:
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






# model = DecisionTreeClassifier(max_depth=4)
# model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20)
# model = DecisionTreeClassifier(max_depth=4, min_samples_split=500, min_samples_leaf=25, random_state=random_state)
# model = RandomForestClassifier(n_estimators=20,min_samples_split=4,min_samples_leaf=2, n_jobs=-1,verbose=True)


# overfit to the training data deliberately so we can see which features it likes the most
# model = DecisionTreeClassifier(max_depth=10)
# model = DecisionTreeClassifier(max_depth=8, min_samples_split=50, min_samples_leaf=40)
model = DecisionTreeClassifier(max_depth=8, min_samples_split=50, min_samples_leaf=40, min_impurity_split=0.15)


# DEBUG
# DEBUG
# DEBUG
X_train = X_test = X_all
y_train = y_test = y_all
# DEBUG
# DEBUG
# DEBUG


# print("fitting decision tree")
model.fit(X_all, y_all)
save_object("decision_tree.pkl", model)

# mean_train_accuracy = model.score(X_train, y_train)
# print("train accuracy: {:.4f}".format(mean_train_accuracy))

print("exporting decision tree visualization")
# feature_names = [str(x) for x in list(range(X_all.shape[1]))]
dot_data = tree.export_graphviz(model, feature_names=feature_names, class_names=class_names, filled=True, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree.pdf")







# some_arbitrary_trial = 0#-13#125#50
# single_window = subject_data[some_arbitrary_trial,:,:]

# x = np.arange(single_window.shape[0])

# fig = plt.figure("Trial #{}".format(some_arbitrary_trial))
# plt.title("Trial #{}".format(some_arbitrary_trial))

# # accelerometer
# plt.scatter(x, single_window[:,0], c='xkcd:red', alpha=0.5, label="accel x")
# plt.scatter(x, single_window[:,1], c='xkcd:orange', alpha=0.5, label="accel y")
# plt.scatter(x, single_window[:,2], c='xkcd:goldenrod', alpha=0.5, label="accel z")

# # gyroscope
# plt.scatter(x, single_window[:,3], c='xkcd:green', alpha=0.5, label="gyro x")
# plt.scatter(x, single_window[:,4], c='xkcd:blue', alpha=0.5, label="gyro y")
# plt.scatter(x, single_window[:,5], c='xkcd:indigo', alpha=0.5, label="gyro z")

# # eog signals
# plt.scatter(x, single_window[:,6], c='xkcd:violet', alpha=0.5, label="eog l")
# plt.scatter(x, single_window[:,7], c='xkcd:gray', alpha=0.5, label="eog r")
# plt.scatter(x, single_window[:,8], c='xkcd:black', alpha=0.5, label="eog h")
# plt.scatter(x, single_window[:,9], c='xkcd:bright pink', alpha=0.5, label="eog v")

# plt.xlabel("Frame #")
# plt.ylabel("Magnitude of feature")
# plt.legend(loc=2)
# plt.show()



