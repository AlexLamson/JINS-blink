import sys
sys.path.append('..')
from util import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

from prepare_data import get_data



X_all, y_all, groups, feature_names, subjects, labels, class_names, is_moving_data, include_eog, include_imu = get_data(use_precomputed=True)







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
save_obj("decision_tree.pkl", model)

# mean_train_accuracy = model.score(X_train, y_train)
# print("train accuracy: {:.4f}".format(mean_train_accuracy))

print("exporting decision tree visualization")
# feature_names = [str(x) for x in list(range(X_all.shape[1]))]
dot_data = export_graphviz(model, feature_names=feature_names, class_names=class_names, filled=True, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree.pdf")
