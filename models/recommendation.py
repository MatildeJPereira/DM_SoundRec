class Recommendation:
    def __init__(self, track_id, data, n_rec=3):
        self.track_id = track_id
        self.data = data
        self.n_rec = n_rec

    def recommend(self):
        data = self.data
        print(data)
        print(self.track_id,' type:', type(self.track_id))
        print(data.index)
        print("Filtered data:", data.loc[data.index == self.track_id])

        find_cluster = data.loc[data.index == self.track_id, 'cluster'].values[0]
        similar_tracks = data[data['cluster'] == find_cluster]

        return similar_tracks.sample(n=self.n_rec)

