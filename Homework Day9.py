import matplotlib.pyplot as plt
import numpy as np
import time

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
