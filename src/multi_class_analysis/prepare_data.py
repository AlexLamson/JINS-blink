import numpy as np
import scipy
from tqdm import tqdm

from util import *
from feature_extractor import get_features



if __name__ == "__main__":
    print("This file isn't meant to be run directly. Run train_machine_learning_model.py instead")


def get_path(subject_id, label_id):
    path = "C:/Data_Experiment_W!NCE/{0}/FACS/label{1}/jins/{0}_label{1}.mat".format(subject_id, label_id)
    return path


def get_data(use_precomputed=False):
    if use_precomputed:
        filename = "all_data.pkl"

        if not isfile(filename):
            print("couldn't load pickle file. recomputing features")
            return get_data(use_precomputed=False)

        else:
            print("loading pickled data")
            (X_all, y_all, groups, feature_names, subject_ids, labels, class_names) = load_object(filename)

            return (X_all, y_all, groups, feature_names, subjects, labels, class_names)

    else:

        subjects = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
        labels = [1, 2, 3, 4, 5]
        class_names = "none,eyebrows lower,eyebrows raiser,cheek raiser,nose wrinkler,upper lip raiser,mouth open".split(',')

        X_all_raw = None
        X_all = None
        y_all = []
        groups = []


        # accumulate the data for the all the subjects
        print("reading raw data into memory")
        for subject in subjects:
            # subject_data = np.zeros(shape=(0,201,10))
            for label in labels:
                path = get_path(subject, label)

                # [ trial * window frames * sensor channels ]
                matlab_object = scipy.io.loadmat(path)

                subject_matrix = matlab_object['data_chunk']
                # subject_data = np.concatenate((subject_data, subject_matrix), axis=0)
                groups += [subject]*subject_matrix.shape[0]
                y_all += [label]*subject_matrix.shape[0]

                for trial in range(subject_matrix.shape[0]):
                    raw_window = subject_matrix[trial,:,:]
                    # print(raw_window.shape)
                    if X_all_raw is None:
                        X_all_raw = np.empty(shape=(0,len(raw_window), 10), dtype=float)
                    # print(X_all_raw.shape)
                    # exit()
                    X_all_raw = np.concatenate((X_all_raw, raw_window[np.newaxis,:,:]), axis=0)


        print("normalizing data")
        # normalize accelerometer signals
        a = np.mean(np.std(X_all_raw[:,:,0:3], axis=2))
        X_all_raw[:,:,0:3] = X_all_raw[:,:,0:3] / a

        # normalize gyroscope signals
        a = np.mean(np.std(X_all_raw[:,:,3:6], axis=2))
        X_all_raw[:,:,3:6] = X_all_raw[:,:,3:6] / a

        # normalize eog signals
        a = np.mean(np.std(X_all_raw[:,:,6:], axis=2))
        X_all_raw[:,:,6:10] = X_all_raw[:,:,6:10] / a


        print("extracting features")
        for trial in tqdm(range(X_all_raw.shape[0])):
            feature_extracted_window, feature_names = get_features(X_all_raw[trial,:,:])
            feature_extracted_window = np.array(feature_extracted_window)

            if X_all is None:
                X_all = np.empty(shape=(0,len(feature_extracted_window)), dtype=float)
            X_all = np.concatenate((X_all, feature_extracted_window[np.newaxis,:]), axis=0)


        y_all = np.array(y_all)
        # np.savetxt("y_all.txt", y_all)  # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
        groups = np.array(groups)

        print("pickling data")
        save_object("all_data.pkl", (X_all, y_all, groups, feature_names, subjects, labels, class_names))

        return (X_all, y_all, groups, feature_names, subjects, labels, class_names)
