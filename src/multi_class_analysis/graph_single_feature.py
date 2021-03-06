import sys
sys.path.append('..')
from util import *

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from prepare_data import get_data


X_all, y_all, groups, feature_names, subjects, labels, class_names = get_data(use_precomputed=True)





try:
    # feature_index = [('90' in x and 'deriv' in x and 'eog h' in x) for x in feature_names].index(True)
    feature_index = [('max of eog v' in x) for x in feature_names].index(True) #super great
    # feature_index = [('min' in x and 'deriv' in x and 'eog l' in x) for x in feature_names].index(True)

    # feature_index = [('std dev of eog h' in x) for x in feature_names].index(True) #meh
    # feature_index = [('std dev of eog v' in x) for x in feature_names].index(True) #quite good
    # feature_index = [('std dev of eog r' in x) for x in feature_names].index(True) #same as previous
    # feature_index = [('std dev of eog l' in x) for x in feature_names].index(True) #same as previous

    # feature_index = [('energy of eog l' in x) for x in feature_names].index(True) #different std devs
    # feature_index = [('max of gyro z' in x) for x in feature_names].index(True) #very bad
except ValueError as e:
    print("Feature not found. Exiting.")
    exit()

print("graphing feature '{}' at index {}".format(feature_names[feature_index], feature_index))



def rand_jitter(arr):
    stdev = 0.1
    return arr + np.random.randn(len(arr)) * stdev



mask = np.where(y_all == labels[0])
for i,label in enumerate(labels):
    mask = np.where(y_all == label)
    x_vals = [label]*X_all[mask,feature_index].T.shape[0]
    y_vals = X_all[mask,feature_index]
    plt.scatter(rand_jitter(x_vals), y_vals, label=class_names[i], alpha=0.1)
plt.xticks(labels, class_names, rotation=-10)
plt.title("'{}' over all windows (+ jitter)".format(feature_names[feature_index]).title())
plt.show()
