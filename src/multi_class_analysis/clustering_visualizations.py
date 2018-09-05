import sys
sys.path.append('..')
from util import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, KMeans

from prepare_data import get_data



X_all, y_all, groups, feature_names, subjects, labels, class_names, is_moving_data, include_eog, include_imu = get_data(use_precomputed=True)


def rand_jitter(arr):
    stdev = 0.05*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev



print("reducing dimensions with PCA")
plt.figure(1)
pca = PCA(n_components=2)
pca.fit(X_all)
X_all_reduced = pca.transform(X_all)
for i,label in enumerate(labels):
    mask = np.where(y_all == label)
    plt.scatter(X_all_reduced[mask,0], X_all_reduced[mask,1], label=class_names[i])
plt.legend(loc=2)
plt.set_cmap('hsv')
plt.title("Data => PCA {}->2".format(X_all.shape[1]))
plt.show()


print("reducing dimensions with t-SNE")
plt.figure(2)
tsne = TSNE(n_components=2)
X_all_reduced = tsne.fit_transform(X_all)
for i,label in enumerate(labels):
    mask = np.where(y_all == label)
    plt.scatter(X_all_reduced[mask,0], X_all_reduced[mask,1], label=class_names[i])
plt.legend(loc=2)
plt.set_cmap('hsv')
plt.title("Dat a=> t-SNE {}->2".format(X_all.shape[1]))
plt.show()


# print("reducing dimensions with GMM then PCA")
# plt.figure(3)
# n_components = 6
# gmm = GaussianMixture(n_components=n_components, covariance_type='tied')
# gmm.fit(X_all)
# X_all_reduced = gmm.predict_proba(X_all)

# pca = PCA(n_components=2)
# pca.fit(X_all_reduced)
# X_all_reduced = pca.transform(X_all_reduced)

# X_all_reduced[:,0] = rand_jitter(X_all_reduced[:,0])
# X_all_reduced[:,1] = rand_jitter(X_all_reduced[:,1])

# for i,label in enumerate(labels):
#     mask = np.where(y_all == label)
#     plt.scatter(X_all_reduced[mask,0], X_all_reduced[mask,1], label=class_names[i])
# plt.legend(loc=2)
# plt.set_cmap('hsv')
# plt.title("data => GMM {0}->{1} => PCA {1}->2".format(X_all.shape[1], n_components))
# plt.show()



print("reducing dimensions with spectral clustering then PCA")
plt.figure(4)
n_components = 7
spectral = SpectralClustering(n_clusters=n_components)
X_all_reduced = spectral.fit_predict(X_all)

# one-hot-ify it
b = np.zeros((X_all_reduced.shape[0], n_components))
b[np.arange(X_all_reduced.shape[0]), X_all_reduced] = 1
X_all_reduced = b

pca = PCA(n_components=2)
pca.fit(X_all_reduced)
X_all_reduced = pca.transform(X_all_reduced)

X_all_reduced[:,0] = rand_jitter(X_all_reduced[:,0])
X_all_reduced[:,1] = rand_jitter(X_all_reduced[:,1])

for i,label in enumerate(labels):
    mask = np.where(y_all == label)
    plt.scatter(X_all_reduced[mask,0], X_all_reduced[mask,1], label=class_names[i])
plt.legend(loc=2)
plt.set_cmap('hsv')
plt.title("data => spectral clustering {0}->{1} => PCA {1}->2".format(X_all.shape[1], n_components))
plt.show()



print("reducing dimensions with k-means clustering then PCA")
plt.figure(5)
n_components = 5
kmeans = KMeans(n_clusters=n_components)
X_all_reduced = kmeans.fit_predict(X_all)

# one-hot-ify it
b = np.zeros((X_all_reduced.shape[0], n_components))
b[np.arange(X_all_reduced.shape[0]), X_all_reduced] = 1
X_all_reduced = b

pca = PCA(n_components=2)
pca.fit(X_all_reduced)
X_all_reduced = pca.transform(X_all_reduced)

X_all_reduced[:,0] = rand_jitter(X_all_reduced[:,0])
X_all_reduced[:,1] = rand_jitter(X_all_reduced[:,1])

for i,label in enumerate(labels):
    mask = np.where(y_all == label)
    plt.scatter(X_all_reduced[mask,0], X_all_reduced[mask,1], label=class_names[i])
plt.legend(loc=2)
plt.set_cmap('hsv')
plt.title("data => k-means clustering {0}->{1} => PCA {1}->2".format(X_all.shape[1], n_components))
plt.show()

