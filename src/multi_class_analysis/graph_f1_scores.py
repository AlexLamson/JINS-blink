from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from util import save_object, load_object
from prepare_data import get_data





# X_all, y_all, groups, feature_names, subjects, labels, class_names, is_moving_data = get_data(use_precomputed=True)


stationary_scores = load_object("f1_scores_stationary.pkl")
mobile_scores = load_object("f1_scores_mobile.pkl")





# import seaborn as sns
# sns.set()

# Plot the F1 scores

# sns.barplot(np.arange(stationary_scores.shape[1]), np.mean(stationary_scores, axis=0))
# sns.barplot(np.arange(mobile_scores.shape[1]), np.mean(mobile_scores, axis=0))

# print(stationary_scores.shape)
# exit()


ax = plt.gca()
ax.yaxis.grid(True)

ax.bar(np.arange(stationary_scores.shape[1])-0.2, np.mean(stationary_scores, axis=0), width=0.4, label='stationary')
ax.bar(np.arange(mobile_scores.shape[1])+0.2, np.mean(mobile_scores, axis=0), width=0.4, label='mobile')

plt.ylim(0.0, 1.0)
plt.yticks(np.arange(11)/10, [str(x) for x in np.arange(11)/10])
plt.title("F1 scores for each label")
plt.xlabel('Label')
ax.legend()
plt.show()



ax = plt.gca()
ax.yaxis.grid(True)

ax.bar(np.arange(stationary_scores.shape[0])-0.2, np.mean(stationary_scores, axis=1), width=0.4, label='stationary')
ax.bar(np.arange(mobile_scores.shape[0])+0.2, np.mean(mobile_scores, axis=1), width=0.4, label='mobile')
plt.ylim(0.0, 1.0)
plt.yticks(np.arange(11)/10, [str(x) for x in np.arange(11)/10])
plt.xticks(np.arange(mobile_scores.shape[0]), [str(x+101) for x in np.arange(mobile_scores.shape[0])])

plt.title("F1 scores for each subject")
plt.xlabel('Subject')
# plt.xticks(np.arange(stationary_scores.shape[0]), class_names[:len(labels)])
ax.legend()
plt.show()


exit()






# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)

# women_means = (25, 32, 34, 20, 25)
# women_std = (3, 5, 2, 3, 3)
# rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)


# # plt.xticks(labels, class_names, rotation=-10)
# # plt.title("'{}' over all windows (+ jitter)".format(feature_names[feature_index]).title())
# plt.show()

print(stationary_scores)

    # tips = np.array([stationary_scores, mobile_scores])

    # print(tips.shape)
    # exit()

headers = ['stationary_scores', 'mobile_scores']

tips = pd.DataFrame(tips, columns=headers)

# tips = pd.DataFrame()
# tips['stationary_scores'] = stationary_scores
# tips['mobile_scores'] = mobile_scores


ax = sns.barplot(x="stationary_scores", y="mobile_scores", data=tips)
plt.show()
