import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def dbscan_cluster(points, min_points=5, epsilon=0.1):
    # Step 1: Display all points
    plt.style.use('dark_background')
    plt.scatter(points[:, 0], points[:, 1], c='red', s=10)
    plt.title("Initial Points")
    plt.show()

    # Step 2: Identify core points
    corePts = np.sum(np.sum((points[None, :, :] - points[:, None, :]) ** 2.0, axis=2) < epsilon ** 2.0,
                     axis=1) >= min_points
    plt.scatter(points[corePts, 0], points[corePts, 1], c='green', s=10)
    plt.scatter(points[~corePts, 0], points[~corePts, 1], c='red', s=10)
    plt.title("Core Points vs Non-Core Points")
    plt.show()

    # Step 3: Identify directly reachable points and noise
    reachable = np.max((np.sum((points[None, corePts, :] - points[:, None, :]) ** 2.0, axis=2) < epsilon ** 2.0),
                       axis=1)
    plt.scatter(points[corePts, 0], points[corePts, 1], c='green', s=10)
    plt.scatter(points[reachable * ~corePts, 0], points[reachable * ~corePts, 1], c="lightblue", s=10)
    plt.scatter(points[~reachable, 0], points[~reachable, 1], c='red', s=10)
    plt.title("Reachable Points vs Noise")
    plt.show()

    # Step 4: Create graph of core points and identify clusters
    p = points[corePts]
    cluster_idx = np.zeros(len(p), dtype=int)

    adj_mat = np.sum((p[None, :, :] - p[:, None, :]) ** 2.0, axis=2) < epsilon ** 2.0
    graph = nx.from_numpy_array(adj_mat)

    clusters = list(nx.connected_components(graph))

    # Assign clusters to core points
    for cluster in range(len(clusters)):
        cluster_idx[list(clusters[cluster])] = cluster

    # Step 5: Assign cluster numbers to reachable points and finalize cluster assignments
    clusters_final = np.full(len(points), fill_value=-1, dtype=int)
    clusters_final[corePts] = cluster_idx

    # Assign cluster numbers to non-core reachable points
    for i in range(len(points)):
        if reachable[i] and not corePts[i]:
            # Assign the cluster number of the closest core point (core points are the center of each cluster)
            nearest_core_idx = np.argmin(np.sum((points[i] - points[corePts]) ** 2, axis=1))
            clusters_final[i] = clusters_final[corePts][nearest_core_idx]

    # Step 6: Display final clusters
    unique_clusters = np.unique(clusters_final)
    plt.style.use('dark_background')
    for cluster in unique_clusters:
        if cluster == -1:  # Noise points
            plt.scatter(points[clusters_final == cluster, 0], points[clusters_final == cluster, 1], c='red', s=10,
                        label=f"Noise")
        else:  # Clustered points
            plt.scatter(points[clusters_final == cluster, 0], points[clusters_final == cluster, 1], s=10,
                        label=f"Cluster {cluster}")

    plt.title("Final DBSCAN Clusters")
    plt.legend()
    plt.show()


def plot_elbow(X, n_neighbors=-1):
    # Default to twice the number of features for n_neighbors
    if n_neighbors < 0:
        n_neighbors = X.shape[1] * 2

    # Fit NearestNeighbors to the data
    nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    nearest_neighbors.fit(X)

    # Get the distances to the nearest neighbors
    distances, _ = nearest_neighbors.kneighbors(X, return_distance=True)

    # Sort the distances
    distances = np.sort(distances[:, -1])

    # Plot the elbow curve
    plt.style.use('dark_background')
    plt.plot(distances)
    plt.title(f'Elbow Curve (n_neighbors={n_neighbors})')
    plt.xlabel('Points')
    plt.ylabel('Distance to Nearest Neighbor')
    plt.show()
