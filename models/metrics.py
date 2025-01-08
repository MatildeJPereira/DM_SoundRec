from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


# TODO Add metrics here
class Metrics:
    """
    A class for evaluating clustering performance using various metrics.

    :Attributes:
        - **data** (:class:`pd.DataFrame`): The dataset, including features and cluster assignments.
    """

    def __init__(self, data):
        """
        Initializes the Metrics object with the dataset that contains clustering results.

        :param data: The dataset that contains both the features and the cluster assignments.
        :type data: :class:`pd.DataFrame`
        """
        self.data = data

    def cluster_cohesion(self):
        """
        Calculates the Silhouette Score for the current clustering.

        :return: The Silhouette Score of the clustering.
        :rtype: :class:`float`
        """
        values_without_cluster = self.data.iloc[:, :-1].values
        score = silhouette_score(values_without_cluster, self.data['cluster'])
        print("Silhouette Score:", score)
        return score

    def davies_bouldin(self):
        """
        Calculates the Davies-Bouldin Index for the clustering.

        :return: The Davies-Bouldin Index.
        :rtype: :class:`float`
        """
        values_without_cluster = self.data.iloc[:, :-1].values
        dbi = davies_bouldin_score(values_without_cluster, self.data['cluster'])
        print("Davies Bouldin Index:", dbi)
        return dbi

    def inertia(self):
        """
        Applies the Elbow Method to determine the optimal number of clusters.

        :return: None, but plots an elbow graph to visualize the inertia change with respect to the number of clusters.
        """
        df_without_cluster = self.data.iloc[:, :-1]

        wcss = []
        for k in range(2, 100):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(df_without_cluster)
            wcss.append(kmeans.inertia_)

        # Plot the Elbow graph
        plt.figure(figsize=(8, 5))
        plt.plot(range(2, 100), wcss, marker='o', linestyle='--', color='b')
        plt.title('The Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')
        plt.show()


