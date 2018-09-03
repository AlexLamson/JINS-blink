import sys
sys.path.append('..')
from util import *

import numpy as np
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt


subjects = [101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117]
labels = [1, 2, 3, 4, 5]
class_names = "None,Brow lower,Brow raiser,Cheek raiser,Nose wrinkler,Lip raiser,Mouth open".split(',')

X_all_raw = None
X_all = None  # np.zeros(shape=(0,201*10))
y_all = []
groups = []




# to change the path, change the function in prepare_data.py
from prepare_data import get_path



# change these values to show data for other subjects/labels/trials
graph_subject = 103
graph_label = 5  # in range [1,5]
graph_trial = 1  # in range [1,20]





# map from (subject, label, subject_trial) -> X_all index
index_dict = dict()
curr_index = 0


# accumulate the data for the all the subjects
print("reading raw data into memory")
for subject in tqdm(subjects):
    # subject_data = np.zeros(shape=(0,201,10))
    for label in labels:
        path = get_path(subject, label)

        # [ trial * window frames * sensor channels ]
        subject_matrix = scipy.io.loadmat(path)['data_chunk']

        for subject_trial in range(subject_matrix.shape[0]):
            index_dict[(subject, label, subject_trial)] = curr_index
            curr_index += 1

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







window_index = index_dict[graph_subject, graph_label, graph_trial-1]
print("plotting window for trial at index {}".format(window_index))
single_window = X_all_raw[window_index,:,:]
x = np.arange(single_window.shape[0])

# set the title of the graph
title = "Subject {} '{}' Trial {}".format(graph_subject, class_names[graph_label-1], graph_trial).title()
fig = plt.figure(title)
plt.title(title)

# graph the accelerometer
plt.scatter(x, single_window[:,0], c='xkcd:red', alpha=0.5, label="accel x")
plt.scatter(x, single_window[:,1], c='xkcd:orange', alpha=0.5, label="accel y")
plt.scatter(x, single_window[:,2], c='xkcd:goldenrod', alpha=0.5, label="accel z")

# graph the gyroscope
plt.scatter(x, single_window[:,3], c='xkcd:green', alpha=0.5, label="gyro x")
plt.scatter(x, single_window[:,4], c='xkcd:blue', alpha=0.5, label="gyro y")
plt.scatter(x, single_window[:,5], c='xkcd:indigo', alpha=0.5, label="gyro z")

# graph the eog signals
plt.scatter(x, single_window[:,6], c='xkcd:violet', alpha=0.5, label="eog l")
plt.scatter(x, single_window[:,7], c='xkcd:gray', alpha=0.5, label="eog r")
plt.scatter(x, single_window[:,8], c='xkcd:black', alpha=0.5, label="eog h")
plt.scatter(x, single_window[:,9], c='xkcd:bright pink', alpha=0.5, label="eog v")

plt.xlabel("Frame #")
plt.ylabel("Magnitude of feature")
plt.legend(loc=2)
plt.show()
