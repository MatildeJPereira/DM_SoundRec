class Recommendation:
    """
    A class that provides recommendations based on clustering for a given track.

    :Attributes:
        - **track_id** (:class:`str` or :class:`int`): The ID of the track for which recommendations are needed.
        - **data** (:class:`pd.DataFrame`): The dataset that contains track information, including track IDs and cluster assignments.
        - **n_rec** (:class:`int`): The number of recommendations to return. Defaults to 3.
    """

    def __init__(self, track_id, data, n_rec=3):
        """
        Initializes the Recommendation object with track ID, dataset, and the number of recommendations.

        :param track_id: The ID of the track for which recommendations are generated.
        :type track_id: :class:`str` or :class:`int`
        :param data: The dataset containing track information, including track IDs and cluster assignments.
        :type data: :class:`pd.DataFrame`
        :param n_rec: The number of recommendations to return. Default is 3.
        :type n_rec: :class:`int`
        """
        self.track_id = track_id
        self.data = data
        self.n_rec = n_rec

    def recommend(self):
        """
        Recommends tracks that are in the same cluster as the given track.

        :return: A DataFrame containing recommended tracks from the same cluster as the input track.
        :rtype: :class:`pd.DataFrame`
        """
        data = self.data

        find_cluster = data.loc[data.index == self.track_id, 'cluster'].values[0]
        similar_tracks = data[data['cluster'] == find_cluster]

        return similar_tracks.sample(n=self.n_rec)

