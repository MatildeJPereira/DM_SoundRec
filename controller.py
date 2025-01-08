import json
from flask import render_template, request, jsonify
from matplotlib import pyplot as plt
import matplotlib
from models.clustering import Clustering
from models.metrics import Metrics
from models.recommendation import Recommendation
from models.cosine_similarity import CosineSimilarity
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
matplotlib.use('Agg')

# Global variable for caching the dataset
cached_data = None


def load_data():
    """
    Load and preprocess the dataset for clustering and analysis.

    The function caches the data to avoid reloading and preprocessing multiple times.

    Steps:
        - Load `raw_echonest.csv` and select key audio features.
        - Load `raw_tracks.csv` to access track metadata.
        - Filter tracks based on common track IDs between the two datasets.
        - Scale selected feature data using StandardScaler.

    :returns: Tuple of (scaled_data DataFrame, filtered_tracks DataFrame).
    :rtype: tuple
    """
    global cached_data
    # check if data is already loaded and cached
    if cached_data is None:
        print("Loading data")
        # Load echonest data taking into account the multi-level headers
        data = pd.read_csv("fma_metadata/raw_echonest.csv", header=[0, 1, 2], index_col=0)
        selected_data = data.loc[:, data.columns.get_level_values(level=2).isin([
            "acousticness", "danceability", "energy", "instrumentalness", "liveness", "speechiness", "tempo", "valence"])]

        selected_data = selected_data.dropna()

        # Load tracks data
        tracks = pd.read_csv('fma_metadata/raw_tracks.csv', header=[0], index_col=0)

        # Filter Tracks
        ids_in_data = selected_data.index
        filtered_tracks = tracks[tracks.index.isin(ids_in_data)]

        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(selected_data)
        scaled_data_df = pd.DataFrame(scaled_data, index=selected_data.index, columns=selected_data.columns)

        cached_data = scaled_data_df, filtered_tracks
    else:
        print("Using cached data")
    return cached_data


def home():
    """
    Renders the home page of the application.
    Fetches track metadata for rendering and passes the data to the template.

    :returns: Rendered home page template.
    :rtype: flask.Response
    """
    data, tracks = load_data()

    return render_template('home.html', songs=tracks)


def update_song():
    """
    Fetch the MP3 URL for the selected song.

    Processes the user input from a JSON POST request and retrieves the MP3 link
    for the selected track based on its track URL.

    :returns: JSON response containing the MP3 URL.
    :rtype: flask.Response
    """
    data, tracks = load_data()
    track_id = int(request.json.get('selectedSong'))

    print("Getting chosen track")
    chosen_track = tracks[tracks.index == track_id]
    chosen_track_url = chosen_track['track_url'].get(track_id)
    chosen_track_mp3 = __get_track_mp3(chosen_track_url)
    print("Chosen track done!")

    return jsonify(chosen_track_mp3)


def process():
    """
    Perform clustering, similarity analysis, and song recommendation.

    Steps:
        - Cluster the dataset using the selected algorithm.
        - Compute similarity between tracks in the same cluster.
        - Recommend top-N similar tracks based on cosine similarity.
        - Generate visualizations and retrieve MP3 URLs for recommendations.

    :returns: JSON response with recommended tracks' metadata and MP3 URLs.
    :rtype: dict
    """
    data, tracks = load_data()
    track_id = int(request.json.get('song'))
    selected_algorithm = request.json.get('algorithm')

    print("Clustering... (", selected_algorithm, ")")
    # # Get the graphs for metrics
    # cc = []
    # db = []
    # for i in range(4, 20):
    #     i = i/10
    #     data_for = data.copy()
    #     clustering = Clustering(data_for, selected_algorithm, 5, i)
    #     data_for = clustering.clustering_alg()
    #
    #     print("Number of clusters:", len(data_for['cluster'].unique()))
    #     # Initialize the metrics
    #     metrics = Metrics(data_for)
    #     # Cluster Cohesion Metric by Silhouette Score
    #     cc.append(metrics.cluster_cohesion())
    #     db.append(metrics.davies_bouldin())
    #     print("***********************\n")
    # plt.plot(db)
    # plt.show()

    clustering = Clustering(data, selected_algorithm, 6, 1.4)
    data = clustering.clustering_alg()
    print("Number of clusters:", len(data['cluster'].unique()))

    # Metrics evaluation
    metrics = Metrics(data)
    metrics.cluster_cohesion()
    metrics.davies_bouldin()

    # Similarity and recommendations
    similarity = CosineSimilarity()
    find_cluster = data.loc[data.index == track_id, 'cluster'].values[0]
    cluster_tracks = data[data["cluster"] == find_cluster]

    track_features = cluster_tracks.loc[track_id].drop(['cluster']).values
    cluster_features = cluster_tracks.drop(['cluster'], axis=1).values
    similarities = similarity.calculate(cluster_features, track_features)

    # Sort
    similar_indices = cluster_tracks.index[np.argsort(similarities)[::-1]]
    rec = cluster_tracks.loc[similar_indices]
    rec = rec[rec.index != track_id].head(5)  # Exclude the input track and get top-N recommendations

    # Recommendation
    print("Loading recommendation")
    rec_tracks = tracks[tracks.index.isin(rec.index)]
    rec_tracks_info = rec_tracks[['artist_name', 'track_title', 'track_url']].copy()

    print("Getting Mp3 for recommended tracks")
    rec_tracks_url = rec_tracks_info['track_url'].tolist()
    rec_tracks_url_mp3 = []
    for track in rec_tracks_url:
        track_mp3 = __get_track_mp3(track)
        rec_tracks_url_mp3.append(track_mp3)
    rec_tracks_info['track_url'] = rec_tracks_url_mp3
    rec_tracks_json = rec_tracks_info.to_dict(orient="index")
    print("Mp3's Done!")

    clustering.show()

    return rec_tracks_json


def __recommend(track_id, data):
    """
    Private method to load the recommendation.

    :param track_id: The track ID for the base recommendation.
    :type track_id: int
    :param data: The dataset used for clustering and recommendations.
    :type data: pandas.DataFrame
    :returns: Recommended tracks.
    :rtype: pandas.DataFrame
    """
    rec = Recommendation(track_id, data).recommend()
    return rec


def __get_track_mp3(track_url):
    """
    Private method to retrieve the MP3 URL of a given track.

    Parses the FMA website for the MP3 file link of the requested track.

    :param track_url: URL of the track on the FMA website.
    :type track_url: str
    :returns: MP3 file URL if available, else None.
    :rtype: str
    """
    res = requests.get(track_url)
    if res.status_code == 200:
        soup = BeautifulSoup(res.content, 'html.parser')
        song_tag = soup.find('div', {'class': 'play-item'})
        if song_tag:
            song_info_str = song_tag['data-track-info']
            song_info = json.loads(song_info_str)
            return song_info['fileUrl']
    return None


def __get_silhouette_analysis(data):
    """
    Private method for silhouette analysis to determine optimal clusters.

    Generates a plot of silhouette scores for k values from 2 to 50.

    :param data: Dataset for clustering analysis.
    :type data: pandas.DataFrame
    """
    scores = []
    for i in range(2, 50):
        c = Clustering(data, 'kmeans', i, 1.4)
        d = c.clustering_alg()
        m = Metrics(d)
        print("k:", i)
        scores.append(m.cluster_cohesion())
    print(max(scores))
    plt.plot(scores)
    plt.show()

