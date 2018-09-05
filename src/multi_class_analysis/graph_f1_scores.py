import sys
sys.path.append('..')
from util import *

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from util import save_obj, load_obj
from prepare_data import get_data



# X_all, y_all, groups, feature_names, subjects, labels, class_names, is_moving_data, include_eog, include_imu = get_data(use_precomputed=True)
class_names = "None,Brow lower,Brow raiser,Cheek raiser,Nose wrinkler".split(',')





# Figure 7 - stationary vs mobile
stationary_scores = load_object("f1_scores/aggregated/f1_scores-stationary-yes_eog-yes_imu.pkl")
mobile_scores = load_object("f1_scores/aggregated/f1_scores-mobile-yes_eog-yes_imu.pkl")

ax = plt.gca()
ax.yaxis.grid(True)

ax.bar(np.arange(5)*4+0, stationary_scores, label='stationary')
ax.bar(np.arange(5)*4+1, mobile_scores, label='mobile')

plt.ylim(0.0, 1.0)
plt.yticks(np.arange(11)/10, [str(x) for x in np.arange(11)/10])
# plt.xticks(np.arange(len(class_names)), class_names)
plt.xticks(np.arange(5)*4+0.5, class_names)
plt.title("F1 scores for each label")
plt.ylabel('F1 Score')
ax.legend()
plt.show()




# Figure 8 - stationary with different data
both_scores = load_object("f1_scores/aggregated/f1_scores-stationary-yes_eog-yes_imu.pkl")
eog_scores = load_object("f1_scores/aggregated/f1_scores-stationary-yes_eog-no_imu.pkl")
imu_scores = load_object("f1_scores/aggregated/f1_scores-stationary-no_eog-yes_imu.pkl")

ax = plt.gca()
ax.yaxis.grid(True)

ax.bar(np.arange(5)*4+0, both_scores, label='EOG+IMU')
ax.bar(np.arange(5)*4+1, imu_scores, label='IMU')
ax.bar(np.arange(5)*4+2, eog_scores, label='EOG')

plt.ylim(0.0, 1.0)
plt.yticks(np.arange(11)/10, [str(x) for x in np.arange(11)/10])
# plt.xticks(np.arange(len(class_names)), class_names)
plt.xticks(np.arange(5)*4+1, class_names)
plt.title("F1 scores for each label")
plt.ylabel('F1 Score')
ax.legend()
plt.show()





# Figure 9 - mobile with different data
both_scores = load_object("f1_scores/aggregated/f1_scores-mobile-yes_eog-yes_imu.pkl")
eog_scores = load_object("f1_scores/aggregated/f1_scores-mobile-yes_eog-no_imu.pkl")
imu_scores = load_object("f1_scores/aggregated/f1_scores-mobile-no_eog-yes_imu.pkl")

ax = plt.gca()
ax.yaxis.grid(True)

ax.bar(np.arange(5)*4+0, both_scores, label='EOG+IMU')
ax.bar(np.arange(5)*4+1, imu_scores, label='IMU')
ax.bar(np.arange(5)*4+2, eog_scores, label='EOG')

plt.ylim(0.0, 1.0)
plt.yticks(np.arange(11)/10, [str(x) for x in np.arange(11)/10])
# plt.xticks(np.arange(len(class_names)), class_names)
plt.xticks(np.arange(5)*4+1, class_names)
plt.title("F1 scores for each label")
plt.ylabel('F1 Score')
ax.legend()
plt.show()









# Figure 10 c - CDF of f1 scores
stationary_scores = np.mean(load_object("f1_scores/f1_scores-stationary-yes_eog-yes_imu.pkl"), axis=1)
mobile_scores = np.mean(load_object("f1_scores/f1_scores-mobile-yes_eog-yes_imu.pkl"), axis=1)

n_bins = 17*2
fig, ax = plt.subplots(figsize=(8, 4))
# plot the cumulative histogram
n, bins, patches = ax.hist(stationary_scores, n_bins, density=True, histtype='step', cumulative=True, label='Stationary')
n, bins, patches = ax.hist(mobile_scores, n_bins, density=True, histtype='step', cumulative=True, label='Mobile')

ax.legend(loc='upper left')
plt.xlabel('F1 Score')
plt.ylabel('CDF')
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.show()







# ax = plt.gca()
# ax.yaxis.grid(True)

# ax.plot(np.arange(5)*4+0, both_scores, label='EOG+IMU')
# ax.plot(np.arange(5)*4+1, imu_scores, label='IMU')

# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# # plt.yticks(np.arange(11)/10, [str(x) for x in np.arange(11)/10])
# # plt.xticks(np.arange(len(class_names)), class_names)
# # plt.xticks(np.arange(5)*4+0.5, class_names)
# plt.title("F1 scores for each label")
# plt.ylabel('F1 Score')
# ax.legend()
# plt.show()













# ax = plt.gca()
# ax.yaxis.grid(True)

# ax.bar(np.arange(stationary_scores.shape[0])-0.2, np.mean(stationary_scores, axis=1), width=0.4, label='stationary')
# ax.bar(np.arange(mobile_scores.shape[0])+0.2, np.mean(mobile_scores, axis=1), width=0.4, label='mobile')
# plt.ylim(0.0, 1.0)
# plt.yticks(np.arange(11)/10, [str(x) for x in np.arange(11)/10])
# plt.xticks(np.arange(mobile_scores.shape[0]), [str(x+101) for x in np.arange(mobile_scores.shape[0])])

# plt.title("F1 scores for each subject")
# plt.xlabel('Subject')
# # plt.xticks(np.arange(stationary_scores.shape[0]), class_names[:len(labels)])
# ax.legend()
# plt.show()

