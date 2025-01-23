import matplotlib.pyplot as plt
import numpy as np
import time
import random
real_centroids, number_of_centroids =  [(np.random.randint(0, 10),np.random.randint(0, 10)),
                                        (np.random.randint(0, 10),np.random.randint(0, 10)),
                                        (np.random.randint(0, 10),np.random.randint(0, 10)),
                                        (np.random.randint(0, 10),np.random.randint(0, 10))], 4

points_per_centroid = 300
all_points = []
colors = ['blue', 'green', 'purple', 'orange']

def generate_point_around_centroid(centroid: tuple) -> tuple:
    x_centroid, y_centroid = centroid
    x = np.random.normal(x_centroid, 1)
    y = np.random.normal(y_centroid, 1)
    return x, y

def distance(p1: tuple, p2: tuple) -> float:
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def mean_centroid(points: list) -> tuple:
    if not points:
        return (0, 0)
    x_sum = (p[0] for p in points)
    y_sum = (p[1] for p in points)
    return (sum(x_sum) / len(points), sum(y_sum) / len(points))

def kmeans_loop(data_points, k, max_iterations=500):
    start_time = time.time()
    centroids = random.sample(data_points, k)
    print("Initial Centroids:", centroids)

    for i in range(max_iterations):
        clusters = [[] for _ in range(k)]

        for point in data_points:
            distances = [distance(point, centroid) for centroid in centroids]
            closest_centroid_index = distances.index(min(distances))
            clusters[closest_centroid_index].append(point)

        new_centroids = [mean_centroid(cluster) for cluster in clusters]

        if new_centroids == centroids:
            print(f"Centroids converged after {i + 1} iterations.")
            break

        centroids = new_centroids

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds.")
    print("Final Centroids:", centroids)
    return clusters, centroids, end_time - start_time

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


for centroid in real_centroids:
    for _ in range(points_per_centroid):
        all_points.append(generate_point_around_centroid(centroid))

plt.figure(figsize=(8, 8))
points_x = [p[0] for p in all_points]
points_y = [p[1] for p in all_points]
plt.scatter(points_x, points_y, s=10, alpha=0.5, label='Generated Points')
centroid_x = [c[0] for c in real_centroids]
centroid_y = [c[1] for c in real_centroids]
plt.scatter(centroid_x, centroid_y, s=100, color='red', marker='x', label='Real Centroids')
plt.title('Generated Data Points and Real Centroids')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig('initial_data.png')
plt.show()
plt.close()

# using Python loop

k_clusters = number_of_centroids
plain_clusters, plain_centroids, plain_time = kmeans_loop(all_points, k_clusters)
print(f"Plain Python k-means computation time: {plain_time:.4f} seconds")

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

for i in range(k_clusters):
    cluster_x = [p[0] for p in plain_clusters[i]]
    cluster_y = [p[1] for p in plain_clusters[i]]
    plt.scatter(cluster_x, cluster_y, s=10, color=colors[i], alpha=0.5, label=f'Cluster {i+1}')

plt.figure(figsize=(8, 8))
centroid_x = [c[0] for c in plain_centroids]
centroid_y = [c[1] for c in plain_centroids]
plt.scatter(centroid_x, centroid_y, s=200, color='red', marker='*', label='K-means Centroids')


plt.title('Clusters and Centroids found by Plain Python k-means')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_plain_clusters.png')
plt.show()
plt.close()

# using umpy
numpy_cluster_indices, numpy_centroids, numpy_time = kmeans_numpy(all_points, k_clusters)

print(f"Vectorized NumPy k-means computation time: {numpy_time} seconds")

plt.figure(figsize=(8, 8))

for i in range(k_clusters):
    cluster_points = np.array(all_points)[numpy_cluster_indices == i]
    cluster_x = cluster_points[:, 0]
    cluster_y = cluster_points[:, 1]
    plt.scatter(cluster_x, cluster_y, s=10, color=colors[i], alpha=0.5, label=f'Cluster {i+1}')

centroid_x = numpy_centroids[:, 0]
centroid_y = numpy_centroids[:, 1]
plt.scatter(centroid_x, centroid_y, s=200, color='red', marker='*', label='K-means Centroids (NumPy)')

plt.title('Clusters and Centroids found by Vectorized NumPy k-means')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_numpy_clusters.png')
plt.show()
plt.close()
