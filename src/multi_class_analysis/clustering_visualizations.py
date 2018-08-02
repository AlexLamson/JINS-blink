import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, KMeans

from prepare_data import get_data



X_all, y_all, groups, feature_names, subjects, labels, class_names = get_data(use_precomputed=True)


def rand_jitter(arr):
    stdev = .05*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev



print("reducing dimensions with PCA")
plt.figure(1)
pca = PCA(n_components=2)
pca.fit(X_all)
X_all_reduced = pca.transform(X_all)
# print(X_all_reduced.shape)
plt.scatter(X_all_reduced[:,0], X_all_reduced[:,1], c=y_all)
plt.set_cmap('hsv')
plt.title("PCA n=2")
# plt.show()



plt.figure(2)
print("reducing dimensions with t-SNE")
tsne = TSNE(n_components=2)
X_all_reduced = tsne.fit_transform(X_all)
# print(X_all_reduced.shape)
plt.scatter(X_all_reduced[:,0], X_all_reduced[:,1], c=y_all)
plt.set_cmap('hsv')
plt.title("t-SNE n=2")
# plt.show()



print("reducing dimensions with GMM then PCA")
plt.figure(3)
n_components = 6
gmm = GaussianMixture(n_components=n_components, covariance_type='tied')
gmm.fit(X_all)
X_all_reduced = gmm.predict_proba(X_all)

pca = PCA(n_components=2)
pca.fit(X_all_reduced)
X_all_reduced = pca.transform(X_all_reduced)

X_all_reduced[:,0] = rand_jitter(X_all_reduced[:,0])
X_all_reduced[:,1] = rand_jitter(X_all_reduced[:,1])

# print(X_all_reduced.shape)
plt.scatter(X_all_reduced[:,0], X_all_reduced[:,1], c=y_all)
plt.set_cmap('hsv')
plt.title("data -> GMM n={} -> PCA n=2".format(n_components))
# plt.show()



n_components = 7
print("reducing dimensions with spectral clustering then PCA")
plt.figure(4)
spectral = SpectralClustering(n_clusters=n_components)
# spectral.fit(X_all)
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

# print(X_all_reduced.shape)
plt.scatter(X_all_reduced[:,0], X_all_reduced[:,1], c=y_all)
plt.set_cmap('hsv')
plt.title("data -> spectral clustering n={} -> PCA n=2".format(n_components))
# plt.show()



n_components = 5
print("reducing dimensions with k-means clustering then PCA")
plt.figure(5)
kmeans = KMeans(n_clusters=n_components)
# kmeans.fit(X_all)
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

# print(X_all_reduced.shape)
plt.scatter(X_all_reduced[:,0], X_all_reduced[:,1], c=y_all)
plt.set_cmap('hsv')
plt.title("data -> k-means clustering n={} -> PCA n=2".format(n_components))
# plt.show()



plt.show()
