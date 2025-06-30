import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

# Dataset where DBSCAN performs good
X1, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# Dataset where DBSCAN struggles
X2, _ = make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)

def apply_clustering(X, title_prefix):
    models = {
        'DBSCAN': DBSCAN(eps=0.3, min_samples=5),
        'KMeans': KMeans(n_clusters=3, random_state=42),
        'Hierarchical': AgglomerativeClustering(n_clusters=3)
    }

    plt.figure(figsize=(12, 4))
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.fit_predict(X)
        plt.subplot(1, 3, i+1)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Set1')
        plt.title(f"{title_prefix} - {name}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()

apply_clustering(X1, "Moons")
apply_clustering(X2, "Blobs")
