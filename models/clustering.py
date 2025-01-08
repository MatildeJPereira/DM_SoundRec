import statistics

import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage


def __find_eps(data, k):
    """
    Determines the optimal epsilon (eps) value for DBSCAN using the k-distance graph.

    :param data:
        The dataset to analyze.
    :type data: :class:`pd.DataFrame`
    :param k:
        The number of nearest neighbors to consider (equivalent to min_pts).
    :type k: :class:`int`

    :return:
        None. Shows the plot for visual inspection.
    """
    # Compute the k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)

    # Sort the distances (k-th neighbor distance for each point)
    distances = np.sort(distances[:, k - 1], axis=0)

    # Plot the k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('k-Distance Graph', fontsize=16)
    plt.xlabel('Data Points (sorted)', fontsize=14)
    plt.ylabel(f'{k}-th Nearest Neighbor Distance', fontsize=14)
    plt.grid()
    plt.show()


def get_neighbors(data, point_index, eps):
    """
    Finds indices of all points within `eps` distance of the given point.
    Used for DBSCAN
    """
    neighbors = []
    for index, point in enumerate(data):
        if np.linalg.norm(data[point_index] - point) <= eps:
            neighbors.append(index)
    return neighbors


def __dendogram(data):
    """
    Generates a dendrogram for Hierarchical clustering.

    :param data:
        The dataset for which the dendrogram will be created.
    :type data: :class:`pd.DataFrame`

    :return:
        None. Displays the dendrogram plot.
    """
    # Compute linkage matrix
    linkage_data = linkage(data, method="ward")

    # Generate and plot the dendogram
    dendrogram(linkage_data)
    plt.show()


def compute_distance_matrix(data, N):
    """
    Compute the symmetric distance matrix for the dataset.
    Used for Hierarchical Clustering

    :param data: Dataset as a numpy array of shape (N, features)
    :return: Lower triangular distance matrix.
    """
    distance_matrix = np.full((N, N), np.inf)  # Initialize with inf for easier min search
    for i in range(N):
        for j in range(i):  # Compute only the part bellow the diagonal of the matrix
            print(j)
            distance_matrix[i][j] = np.linalg.norm(data[i] - data[j])  # Euclidean distance
    return distance_matrix


def update_distance_matrix(distance_matrix, cluster_a, cluster_b):
    """
    Update the distance matrix after merging two clusters.
    Used for Hierarchical Clustering

    :param distance_matrix: Current distance matrix.
    :param cluster_a: Index of the first cluster.
    :param cluster_b: Index of the second cluster.
    :return: Updated distance matrix.
    """
    for i in range(len(distance_matrix)):
        if i != cluster_a and i != cluster_b:
            # update distance of new merged cluster
            distance_matrix[cluster_a][i] = distance_matrix[i][cluster_a] = min(
                distance_matrix[cluster_a][i], distance_matrix[cluster_b][i]
            )
    # Remove cluster_b by setting its distances to infinity (effectively deleting it)
    distance_matrix[:, cluster_b] = np.inf
    distance_matrix[cluster_b, :] = np.inf
    return distance_matrix


class Clustering:
    """
    Class that creates the chosen clustering algorithm.

    :Attributes:
        - **data** (:class:`pd.DataFrame`):
          The data from the chosen dataset.
        - **clustering_type** (:class:`str`):
          The clustering type that has been chosen ('kmeans', 'dbscan', 'hierarchical').
        - **n_clusters** (:class:`int`):
          The number of clusters for K-Means and hierarchical clustering.
        - **eps** (:class:`float`):
          The epsilon value for DBSCAN.
    """

    def __init__(self, data, clustering_type, n_clusters, eps):
        """
        Initializes the Clustering object with dataset, algorithm type, and parameters.

        :param data:
            The dataset to be used for clustering.
        :type data: :class:`pd.DataFrame`
        :param clustering_type:
            The type of clustering algorithm to apply. Can be one of:
            `'kmeans'`, `'dbscan'`, `'hierarchical'`.
        :type clustering_type: :class:`str`
        :param n_clusters:
            Number of clusters to form for K-Means and hierarchical clustering.
        :type n_clusters: :class:`int`
        :param eps:
            Epsilon parameter for DBSCAN. Defines neighborhood size.
        :type eps: :class:`float`
        """
        self.data = data
        self.type = clustering_type
        self.n_clusters = n_clusters
        self.eps = eps

    def clustering_alg(self):
        """
        Entry point for executing the chosen clustering algorithm.

        :return:
            Data with an additional 'cluster' column containing cluster assignments.
        :rtype: :class:`pd.DataFrame`
        """
        if self.type == 'kmeans':
            return self.__my_kmeans()
        if self.type == 'dbscan':
            return self.__my_dbscan()
        if self.type == 'hierarchical':
            return self.__my_hierarchical()

    def __k_means(self):
        """
        Implements the K-Means clustering algorithm.

        :return:
            Data with an additional 'cluster' column for cluster assignments.
        :rtype: :class:`pd.DataFrame`
        """
        # Create and fit the K-Means model
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.data['cluster'] = kmeans.fit_predict(self.data)
        return self.data

    def __my_kmeans(self):
        """
        Handmade implementation of K-Means

        :return:
            Data with additional 'cluster' column for cluster assigments.
        :rtype: :class:`pd.DataFrame`
        """
        # Initialize centroids randomly from the data points
        np.random.seed(2)
        initial_indices = np.random.choice(len(self.data), self.n_clusters, replace=False)
        centroids = self.data.iloc[initial_indices].to_numpy()

        # Iteratively update centroids and cluster assignments
        for r in range(100):  # Max iterations 100 for reassurance
            clusters = [[] for _ in range(self.n_clusters)]
            for point in self.data.to_numpy():
                distances = [np.linalg.norm(point - centroid) for centroid in centroids]
                closest_centroid_index = distances.index(min(distances))
                clusters[closest_centroid_index].append(point)

            # save the previous centroids
            prev_centroids = centroids.copy()

            # recalculate centroids
            for i, cluster in enumerate(clusters):
                if cluster:
                    centroids[i] = np.mean(cluster, axis=0)

            if np.allclose(centroids, prev_centroids):
                print("Kmeans converged on ", r, "iterations")
                break

        # Add cluster assignments to the dataframe
        cluster_assignments = []
        for point in self.data.to_numpy():
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_assignments.append(np.argmin(distances))

        self.data['cluster'] = cluster_assignments
        return self.data

    def __db_scan(self):
        """
        TODO verify
        TODO __find_eps gives 2.2, trying 0.65, verify more
        Implements the DBSCAN clustering algorithm.

        :return:
            Data with an additional 'cluster' column for cluster assignments.
        :rtype: :class:`pd.DataFrame`
        """
        # Determinge a min_pts (minimum samples) for DBSCAN
        min_pts = self.data.shape[1] * 2
        print("DBSCAN min_pts: ", min_pts)

        # Find a suitable eps
        # self.__find_eps(self.data, min_pts)

        # Apply the DBSCAN algorithm with specified eps and min_samples
        dbscan = DBSCAN(eps=self.eps, min_samples=min_pts)
        self.data['cluster'] = dbscan.fit_predict(self.data)
        return self.data

    def __my_dbscan(self):
        """
        Handmade implementation of the DBSCAN clustering algorithm.

        :return:
            Data with an additional 'cluster' column for cluster assignments.
        :rtype: :class:`pd.DataFrame`
        """
        # Determining a min_pts (minimum samples)
        min_pts = self.data.shape[1] * 2  # Can be adjusted as needed
        print("DBSCAN min_pts: ", min_pts)

        # Initialize variables
        data_np = self.data.to_numpy()
        n_points = len(data_np)
        labels = [-1] * n_points  # all points start as -1, also defines noise
        visited = [False] * n_points
        cluster_id = 0  # cluster starts at 0, increments

        # Perform DBSCAN
        for point_index in range(n_points):
            if visited[point_index]:  # Already visited, skip
                continue

            # mark point as visited
            visited[point_index] = True
            # get neighbors for the current point
            neighbors = get_neighbors(data_np, point_index, self.eps)

            if len(neighbors) < min_pts:
                # mark as noise if not enough neighbors
                labels[point_index] = -1
            else:
                cluster_id += 1
                labels[point_index] = cluster_id

                i = 0
                while i < len(neighbors):
                    neighbor_index = neighbors[i]

                    if not visited[neighbor_index]: # if the neighbor isn't visited
                        # change neighbor to visited
                        visited[neighbor_index] = True
                        # get the new neighbors of the neighbor
                        new_neighbors = get_neighbors(data_np,neighbor_index,self.eps)

                        # check if there are more new neighbors than the minimum points criteria
                        if len(new_neighbors) >= min_pts:
                            neighbors += [n for n in new_neighbors if n not in neighbors]

                    # add neighbor to cluster if it's not already assigned
                    if labels[neighbor_index] == -1:
                        labels[neighbor_index] = cluster_id

                    i += 1

        # Assign labels to the dataframe
        self.data['cluster'] = labels
        return self.data

    def __hierarchical(self):
        """
        Implements the Hierarchical clustering algorithm.

        :return:
            Data with an additional 'cluster' column for cluster assignments.
        :rtype: :class:`pd.DataFrame`
        """
        # Make a dendogram
        # self.__dendogram(self.data)
        # Dendogram Result: number of clusters 5

        # Apply Agglomerative Clustering with the Ward linkage method
        hierarchical_cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')
        self.data['cluster'] = hierarchical_cluster.fit_predict(self.data)

        return self.data

    def __my_hierarchical(self):
        """
        Handmade implementation of Hierarchical Clustering.

        :return:
            Data with additional 'cluster' column for cluster assignments.
        :rtype: :class:`pd.DataFrame`
        """

        # prepare data
        data = self.data.to_numpy()
        N = len(data)
        distance_matrix = compute_distance_matrix(data, N)

        # initial clusters
        clusters = [[i] for i in range(N)]

        # perform Agglomerative Clustering
        while len(clusters) > 1:
            print(len(clusters))
            # Find the two closest clusters
            min_dist = np.inf
            i, j = -1, -1
            for row in range(N):
                for col in range(row):  # Only look at the lower triangle
                    if distance_matrix[row][col] < min_dist:
                        min_dist = distance_matrix[row][col]
                        i, j = row, col

            # join clusters j to i
            clusters[i].extend(clusters[j])
            # remove the cluster j
            print(j)
            clusters.pop(j)


            # update distance matrix
            distance_matrix = update_distance_matrix(distance_matrix, i, j)

        # Assign final cluster labels
        labels = [-1] * N  # Initialize all points as unassigned
        for cluster_id, cluster_points in enumerate(clusters[0]):
            for point in cluster_points:
                labels[point] = cluster_id

        # Add labels to DataFrame
        self.data['cluster'] = labels
        return self.data

    def show(self):
        """
        Visualizes the clustered data using PCA to reduce dimensions.

        :return:
            The file path of the saved scatter plot.
        :rtype: :class:`str`
        """
        # Clear any previous plots
        plt.clf()

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.data)

        # Create a scatter plot
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.data['cluster'], cmap='viridis')
        plt.title('Clustering')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)

        # Save the plot to a file
        plot_path = 'static/clustering_plot.png'
        plt.savefig(plot_path)
        plt.close()

        return plot_path
