from sklearn.datasets import make_blobs
from KMean import KMeansScratch

# Create synthetic data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)

# Apply KMeans
model = KMeansScratch(k=3)
model.fit(X)
model.plot_clusters(X)
