import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Clustering:

    def __init__(self, data, clustering_type, n_clusters):
        self.data = data
        self.type = clustering_type
        self.n_clusters = n_clusters

    def clustering_alg(self):
        """
        TODO this needs to be changed later
        This is the entry point for the clustering, it starts the type of clustering that is selected
        :return: for now it returns the path of the plot created
        """
        if self.type == 'kmeans':
            return self.__k_means()
        if self.type == 'dbscan':
            return self.__db_scan()
        if self.type == 'hierarchical':
            return self.__hierarchical()

    def __k_means(self):
        """
        TODO: implement my own k-means
        This is the k-means clustering algorihtm
        :return: the original data with clustering
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.data['cluster'] = kmeans.fit_predict(self.data)
        return self.data

    def __db_scan(self):
        """
        TODO implement my own dbscan
        TODO is horrible for large datasets, will kill your commputer, DO NOT RUN (yet)
        This is the DBSCAN clustering algorithm
        :return: the data with clustering
        """
        # Find a stuitable min_pts
        min_pts = self.scaled_data.shape[1]*2

        # Find a suitable eps
        # self.__find_eps(self.scaled_data, min_pts)
        eps = 30

        dbscan = DBSCAN(eps=eps, min_samples=min_pts)
        self.data['cluster'] = dbscan.fit_predict(self.scaled_data)
        return self.data

    def __hierarchical(self):
        """
        TODO implement my own hierarchical clustering
        This is the hierarchical clustering algorithm
        :return: nothing yet
        """
        return None

    def show(self):
        """
        This method creates a visual representation of the clustering.
        :return: the path to the plot created
        """
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.scaled_data)

        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.data['cluster'], cmap='viridis')

        plt.title('Clustering')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_path = 'static/clustering_plot.png'
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    def __find_eps(self, data, k):
        """
        Function to find the best eps for the given data for dbscan
        :param data: the data in question
        :param k: the same as min_pts
        :return: nothing, it shows the graph in which one can look for the slope and use its value
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
