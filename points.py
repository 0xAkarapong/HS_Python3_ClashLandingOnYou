import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

colors = ['blue', 'green', 'purple', 'orange']
def kmeans_numpy(data_points, k, max_iterations=500):
    start_time = time.time()
    data_np = np.array(data_points)
    initial_centroid_indices = np.random.choice(len(data_np), k, replace=False)
    centroids_np = data_np[initial_centroid_indices]

    for iteration in range(max_iterations):
        distances = np.sqrt(((data_np[:, np.newaxis, :] - centroids_np[np.newaxis, :, :]) ** 2).sum(axis=2))
        cluster_indices = np.argmin(distances, axis=1)
        new_centroids_np = np.array([data_np[cluster_indices == i].mean(axis=0) if np.any(cluster_indices == i) else centroids_np[i] for i in range(k)])
        if np.all(np.sqrt(np.sum((centroids_np - new_centroids_np)**2, axis=1)) < 1e-4):
            break
        centroids_np = new_centroids_np

    end_time = time.time()
    return cluster_indices, centroids_np, end_time - start_time

data = pd.read_csv('points_example.csv', header=None, names=['x', 'y'])
numpy_cluster_indices, numpy_centroids, numpy_time = kmeans_numpy(data, 4)

print(f"Vectorized NumPy k-means computation time: {numpy_time} seconds")

plt.figure(figsize=(8, 8))

for i in range(4):
    cluster_points = np.array(data)[numpy_cluster_indices == i]
    cluster_x = cluster_points[:, 0]
    cluster_y = cluster_points[:, 1]
    plt.scatter(cluster_x, cluster_y, s=10, color=colors[i], alpha=0.5, label=f'Cluster {i+1}')

centroid_x = numpy_centroids[:, 0]
centroid_y = numpy_centroids[:, 1]
plt.scatter(centroid_x, centroid_y, s=200, color='red', marker='*', label='K-means Centroids (NumPy)')
for i in range(len(centroid_x)):
    print(centroid_x[i], centroid_y[i])
plt.title('Clusters and Centroids found by Vectorized NumPy k-means')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig('points.png')
plt.show()
plt.close()