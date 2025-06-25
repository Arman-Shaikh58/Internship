import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3):
        self.k = k
        self.centroid = None

    @staticmethod
    def _euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iter=200):
        # Initialize centroids randomly within the data range
        self.centroid = np.random.uniform(
            np.amin(X, axis=0),
            np.amax(X, axis=0),
            size=(self.k, X.shape[1])
        )

        for _ in range(max_iter):
            y = []

            # Assign each point to the nearest centroid
            for data_point in X:
                distances = KMeans._euclidean_distance(data_point, self.centroid)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)  # convert list to NumPy array for indexing

            # Recompute centroids
            cluster_centers = []
            for i in range(self.k):
                indices = np.argwhere(y == i).flatten()
                if len(indices) == 0:
                    cluster_centers.append(self.centroid[i])  # keep old centroid if no points
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0))

            cluster_centers = np.array(cluster_centers)

            # Check for convergence
            if np.max(np.abs(self.centroid - cluster_centers)) < 1e-4:
                break
            else:
                self.centroid = cluster_centers

        return y

# Generate random data
random_points = np.random.randint(0, 100, (100, 2))

# Apply KMeans
kmeans = KMeans(k=3)
labels = kmeans.fit(random_points)

# Plot result
plt.scatter(random_points[:, 0], random_points[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.centroid[:, 0], kmeans.centroid[:, 1], c='red', marker="*", s=200)
plt.title("K-Means Clustering (From Scratch)")
plt.show()
