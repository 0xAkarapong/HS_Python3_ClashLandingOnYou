import csv
import matplotlib.pyplot as plt
import numpy as np

def read_points_from_csv(filename):
    points = []
    try:
        with open(filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                try:
                    x = float(row[0])
                    y = float(row[1])
                    points.append([x, y])
                except (ValueError, IndexError):
                    print(f"Skipping invalid row: {row}")
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None
    return points

def find_bounds(points):
    if not points:
        return None, None, None, None
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    return min_x, max_x, min_y, max_y

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def manual_guess_clusters(points, initial_centroids, max_iterations=10):
    if not points or not initial_centroids:
        return np.array([]), np.array([])

    n_clusters = len(initial_centroids)
    cluster_labels = np.zeros(len(points), dtype=int)
    centroids = np.array(initial_centroids)

    for _ in range(max_iterations):
        updated_cluster_labels = np.zeros(len(points), dtype=int)
        for idx, point in enumerate(points):
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            updated_cluster_labels[idx] = np.argmin(distances)

        if np.array_equal(cluster_labels, updated_cluster_labels):
            break # Convergence

        cluster_labels = updated_cluster_labels
        new_centroids = []
        for cluster_id in range(n_clusters):
            cluster_points = [points[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            if cluster_points:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(centroids[cluster_id]) # Keep old centroid if cluster is empty
        centroids = np.array(new_centroids)

    return cluster_labels, centroids


def plot_points_clusters(points, cluster_labels=None, centroids=None, bounds=None):
    plt.figure(figsize=(10, 8))

    point_array = np.array(points)
    if cluster_labels is not None:
        plt.scatter(point_array[:, 0], point_array[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, label='Points in Clusters')
    else:
        plt.scatter(point_array[:, 0], point_array[:, 1], alpha=0.7, label='Points')

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='red', label='Centroids')

    if bounds:
        min_x, max_x, min_y, max_y = bounds
        plt.hlines(y=min_y, xmin=min_x, xmax=max_x, colors='blue', linestyles='dashed', label='Bounds')
        plt.hlines(y=max_y, xmin=min_x, xmax=max_x, colors='blue', linestyles='dashed')
        plt.vlines(x=min_x, ymin=min_y, ymax=max_y, colors='blue', linestyles='dashed')
        plt.vlines(x=max_x, ymin=min_y, ymax=max_y, colors='blue', linestyles='dashed')

    plt.title('Points, Manually Guessed Clusters, and Bounds')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    filename = 'points_example.csv'

    points = read_points_from_csv(filename)

    if points:
        bounds = find_bounds(points)
        print(f"Bounds: X=({bounds[0]:.2f}, {bounds[1]:.2f}), Y=({bounds[2]:.2f}, {bounds[3]:.2f})")


        initial_centroids_guess = [
            points[0],          #
            points[len(points) // 3],
            points[2 * len(points) // 3],
            points[-1]
        ] if len(points) >= 4 else points[:4]

        cluster_labels, centroids = manual_guess_clusters(points, initial_centroids_guess)
        print(f"Manually guessed 4 clusters with initial centroids:\n{initial_centroids_guess}\nand final centroids:\n{centroids}")

        plot_points_clusters(points, cluster_labels, centroids, bounds)
    else:
        print("No points were loaded. Please check the CSV file.")