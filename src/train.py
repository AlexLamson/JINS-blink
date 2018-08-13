from util import *

import numpy as np
from sklearn.model_selection import train_test_split

from aggregate_all_data import get_all_data
from extract_features import extract_features_from_dataframe
from evaluate_classification import evaluate_model as fit_and_evaluate_classifier
from evaluate_regression import evaluate_model as fit_and_evaluate_regressor

from models import classification_decision_tree


def split_data(data_split_sizes, inputs, outputs):
    data_split_sizes = [1.0*x/sum(data_split_sizes) for x in data_split_sizes]
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=data_split_sizes[1])
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    print("aggregating data from all the different trials (NOTE: this is currently not implemented)")
    all_data = get_all_data('../res/')


    # extract features and split into inputs and outputs
    use_precomputed_features = True
    window_size = 10
    features_pickle_filename = data_folder+"features.pickle"
    if use_precomputed_features and file_exists(features_pickle_filename):
        # cache the computation to make it faster in the future
        print('loading precomputed features')
        inputs, outputs = load_obj(features_pickle_filename)
    else:
        print('extracting features & saving to file')
        inputs, outputs = extract_features_from_dataframe(all_data, window_size)
        save_obj((inputs, outputs), features_pickle_filename)


    # uncomment this to sample it so the models train faster
    print("sampling (time-based) less data for faster training")
    seconds_of_data = 60*30
    inputs, outputs = inputs[:seconds_of_data*100], outputs[:seconds_of_data*100]

    # sample blinks more often than open eyes to address class imbalance
    def should_sample(OF_blink_confidence):
        if OF_blink_confidence < 0.025:
            return np.random.rand() < 0.11
        return True
    print("sampling (confidence-based) less data for faster training")
    samples_mask = np.array([should_sample(OF_blink_confidence) for OF_blink_confidence in outputs])
    inputs = inputs[samples_mask]
    outputs = outputs[samples_mask]


    print("splitting data into train and test sets")
    OF_classification_threshold = 0.6
    data_split_sizes = [80, 10, 10]  # train, test, validate
    X_train_r, X_test_r, y_train_r, y_test_r = split_data(data_split_sizes, inputs, outputs)
    X_train_c, X_test_c, y_train_c, y_test_c = X_train_r, X_test_r, y_train_r > OF_classification_threshold, y_test_r > OF_classification_threshold
    outputs_c = outputs > OF_classification_threshold


    """
    Classification models
    NOTE the format of the code below this line should change
    so the models are distributed into different files for readability
    """
    model = classification_decision_tree.get_model()

    print("fitting model & performing cross validation")
    fitness, model = fit_and_evaluate_classifier(model, X_train_c, y_train_c)




    # TODO optimization over hyperparameters (preferably stochastic search as discussed with Chris)


    # save the model
    print('fitting whole dataset & saving model')
    model.fit(inputs, outputs_c)  # be sure to re-train the best model on the full data-set
    save_obj(model, 'model.pickle')
