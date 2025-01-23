import matplotlib.pyplot as plt
import numpy as np

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
        # not finished yet