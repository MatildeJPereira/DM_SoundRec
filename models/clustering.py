import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class Clustering:

    def __init__(self, data, clustering_type, n_clusters):
        # TODO: I can select specific groups of features
        # Use standard scaler to normalize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        self.data = data
        self.scaled_data = scaled_data
        self.type = clustering_type
        self.n_clusters = n_clusters

    def clustering_alg(self):
        """
        TODO this needs to be changed later
        This is the entry point for the clustering, it starts the type of clustering that is selected
        :return: for now it returns the path of the plot created
        """
        if self.type == 'kmeans':
            return self.k_means()
        if self.type == 'dbscan':
            return self.db_scan()
        if self.type == 'hierarchical':
            return self.hierarchical()

    def k_means(self):
        """
        TODO: implement my own k-means
        This is the k-means clustering algorihtm
        :return: for now the original data with clustering
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.data['cluster'] = kmeans.fit_predict(self.scaled_data)
        return self.data

    def db_scan(self):
        """
        TODO implement my own dbscan
        This is the DBSCAN clustering algorithm
        :return: still nothing
        """
        return None

    def hierarchical(self):
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

        plt.title('K-Means Clustering')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Compoenent 2')
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_path = 'static/clustering_plot.png'
        plt.savefig(plot_path)
        plt.close()

        return plot_path
