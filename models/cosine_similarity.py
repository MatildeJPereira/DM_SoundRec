from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity:
    @staticmethod
    def calculate(features, track_features):
        similarities = cosine_similarity(features, track_features.reshape(1,-1)).flatten()
        return similarities
