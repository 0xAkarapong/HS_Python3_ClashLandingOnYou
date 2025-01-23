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
    centroids = random.sample(data_points, k)
    print("Initial Centroids:", centroids)

    start_time = time.time()

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

k_clusters = number_of_centroids
plain_clusters, plain_centroids, plain_time = kmeans_loop(all_points, k_clusters)
print(f"Plain Python k-means computation time: {plain_time:.4f} seconds")

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


