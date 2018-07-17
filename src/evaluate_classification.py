from util import *

import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def evaluate_model(model, inputs, outputs, verbose=False):
    cv = KFold(n_splits=10, shuffle=True, random_state=rand_seed)

    # used for averaging the confusion matrices, fscores
    conf_array = []
    f_array = []
    loss_array = []

    for i, (train_indices, test_indices) in enumerate(cv.split(inputs)):
        # split into training and testing
        inputs_train = inputs[train_indices, :]
        outputs_train = outputs[train_indices]
        inputs_test = inputs[test_indices, :]
        outputs_test = outputs[test_indices]

        model.fit(inputs_train, outputs_train)

        predictions = model.predict(inputs_test)

        conf_array += [ confusion_matrix(outputs_test, predictions) ]
        _, _, fscore, _ = precision_recall_fscore_support(outputs_test, predictions)
        f_array += [ fscore ]

    # suppress threaded debug output printing several seconds after model finishes training
    model.verbose = False

    print('predicted\n open\tblink')
    print( np.mean(conf_array, axis=0) )
    fscore = np.mean(f_array, axis=0)
    print('fscore', fscore)

    # return the model quality score and the fitted model
    return fscore, model
