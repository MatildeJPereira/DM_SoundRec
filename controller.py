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

# cache for storing the data, so it doesn't have to load multiple times
cached_data = None


def load_data():
    global cached_data
    # check if data is already loaded and cached
    if cached_data is None:
        print("Loading data")
        # load the echonest data taking into account the multi-level headers, the track_id is the index
        data = pd.read_csv("fma_metadata/raw_echonest.csv", header=[0, 1, 2], index_col=0)
        selected_data = data.loc[:, data.columns.get_level_values(level=2).isin([
            "acousticness", "danceability", "energy", "instrumentalness", "liveness", "speechiness", "tempo", "valence"])]

        selected_data = selected_data.dropna()

        # load raw tracks data, the track_id is the index
        tracks = pd.read_csv('fma_metadata/raw_tracks.csv', header=[0], index_col=0)

        # choose just the tracks that are in echonest
        ids_in_data = selected_data.index
        filtered_tracks = tracks[tracks.index.isin(ids_in_data)]

        # scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(selected_data)
        scaled_data_df = pd.DataFrame(scaled_data, index=selected_data.index, columns=selected_data.columns)

        cached_data = scaled_data_df, filtered_tracks
    else:
        print("Using cached data")
    return cached_data


def home():
    """
    Home Controller
    :return: Renders the template for the home page with the needed variables
    """
    data, tracks = load_data()
    songs = tracks['track_title']

    return render_template('home.html', songs=tracks)
                           # chosen_track=chosen_track_mp3,
                           # rec=rec_tracks_url,
                           # rec_mp3=rec_tracks_url_mp3,
                           # songs=songs_names,
                           # plot_path=plot_path)


def update_song():
    data, tracks = load_data()
    track_id = int(request.json.get('selectedSong'))

    print("Getting chosen track")
    chosen_track = tracks[tracks.index == track_id]
    chosen_track_url = chosen_track['track_url'].get(track_id)
    chosen_track_mp3 = __get_track_mp3(chosen_track_url)
    print("Chosen track done!")

    return jsonify(chosen_track_mp3)


def process():
    data, tracks = load_data()
    track_id = int(request.json.get('song'))
    selected_algorithm = request.json.get('algorithm')

    # TODO inertia elbow method for each clustering algorithm
    print("Clustering... (", selected_algorithm, ")")
    # for i in range(50, 130):
    #     data_for = data
    #     i = i/100
    #     clustering = Clustering(data_for, selected_algorithm, 8, i)
    #     data_for = clustering.clustering_alg()
    #
    #     print("Number of clusters:", len(data_for['cluster'].unique()))
    #     # Initialize the metrics
    #     metrics = Metrics(data_for)
    #     # Cluster Cohesion Metric by Silhouette Score
    #     metrics.cluster_cohesion()
    #     metrics.davies_bouldin()
    #     print("***********************\n")

    clustering = Clustering(data, selected_algorithm, 8, 0.65)
    data = clustering.clustering_alg()

    # print("Clustering Finished, number of clusters: ", )
    data.to_csv("./static/clustering")

    # Initialize the metrics
    metrics = Metrics(data)
    # Cluster Cohesion Metric by Silhouette Score
    metrics.cluster_cohesion()
    metrics.davies_bouldin()

    # TODO Remove, Temporary to make a silhouette analysis and get the best number for k
    # __get_silhouette_analysis(data)
    # metrics.inertia()

    # get all tracks of the same cluster
    similarity = CosineSimilarity()
    find_cluster = data.loc[data.index == track_id, 'cluster'].values[0]
    cluster_tracks = data[data["cluster"] == find_cluster]

    track_features = cluster_tracks.loc[track_id].drop(['cluster']).values
    cluster_features = cluster_tracks.drop(['cluster'], axis=1).values

    similarities = similarity.calculate(cluster_features, track_features)

    # sort
    similar_indices = cluster_tracks.index[np.argsort(similarities)[::-1]]
    rec = cluster_tracks.loc[similar_indices]
    rec = rec[rec.index != track_id].head(5)  # Exclude the input track and get top-N recommendations
    print(rec)

    # Recommendation
    print("Loading recommendation")
    # rec = __recommend(track_id, data)

    rec_tracks = tracks[tracks.index.isin(rec.index)]
    rec_tracks_info = rec_tracks[['artist_name', 'track_title', 'track_url']].copy()
    rec_tracks_url = rec_tracks_info['track_url'].tolist()
    print("Recommendation finished")

    print("Getting Mp3 for recommended tracks, this can take some time")
    rec_tracks_url_mp3 = []
    for track in rec_tracks_url:
        track_mp3 = __get_track_mp3(track)
        rec_tracks_url_mp3.append(track_mp3)
    rec_tracks_info['track_url'] = rec_tracks_url_mp3
    rec_tracks_json = rec_tracks_info.to_dict(orient="index")
    print("Mp3's Done!")

    plot_path = clustering.show()

    return rec_tracks_json


def __recommend(track_id, data):
    """
    Private method to load the recommendation
    :param track_id: the track in which the recommendation is based on
    :param data: the data
    :return: the recommended tracks, 3 by default
    """
    rec = Recommendation(track_id, data).recommend()
    return rec


def __get_track_mp3(track_url):
    """
    Private Method that parses the FMA website for the song in question
    This way it's possible to play the mp3 on the website
    :param track_url: the url of the track
    :return: if the FMA url exists, returns the mp3 url
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
    scores = []
    for i in range(2, 50):
        c = Clustering(data, 'kmeans', i)
        d = c.clustering_alg()
        m = Metrics(d)
        print("k:", i)
        scores.append(m.cluster_cohesion())
    print(max(scores))
    plt.plot(scores)
    plt.show()

