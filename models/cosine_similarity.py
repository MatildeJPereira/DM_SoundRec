from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity:
    """
    A class to calculate cosine similarity between two sets of features.

    This class uses the cosine similarity metric to measure the similarity between two feature sets:
    the query features and the features of individual tracks or items.

    :Methods:
        - **calculate(features, track_features)**:
        Calculates the cosine similarity between a set of features and a track's features.
    """

    @staticmethod
    def calculate(features, track_features):
        """
        Calculates the cosine similarity between the provided features and the track features.

        :param features: The set of features to compare against.
        :type features: :class:`np.ndarray` or :class:`pd.DataFrame`
        :param track_features: The individual features of a track that we want to measure similarity with.
        :type track_features: :class:`np.ndarray`

        :return: A list of similarity scores between the input features and the track features.
        :rtype: :class:`np.ndarray`
        """

        similarities = cosine_similarity(features, track_features.reshape(1,-1)).flatten()
        return similarities
