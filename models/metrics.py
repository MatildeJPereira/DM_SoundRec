from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


# TODO Add metrics here
class Metrics:
    def __init__(self, data):
        self.data = data

    def cluster_cohesion(self):
        """
        Silhouette Score of the clustering done
        :return: the Silhouette Score
        """
        values_without_cluster = self.data.iloc[:, :-1].values
        score = silhouette_score(values_without_cluster, self.data['cluster'])
        print("Silhouette Score:", score)
        return score

    def davies_bouldin(self):
        values_without_cluster = self.data.iloc[:, :-1].values
        dbi = davies_bouldin_score(values_without_cluster, self.data['cluster'])
        # print(values_without_cluster)
        print("Davies Bouldin Index:", dbi)

    def inertia(self):
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


