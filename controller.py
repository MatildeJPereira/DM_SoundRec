import json

from flask import render_template, request, jsonify
from matplotlib import pyplot as plt

from models.clustering import Clustering
from models.metrics import Metrics
from models.recommendation import Recommendation
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler


def load_data():
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

    # TODO Join the two tables?
    # scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)
    scaled_data_df = pd.DataFrame(scaled_data, index=selected_data.index, columns=selected_data.columns)

    return scaled_data_df, filtered_tracks


def home():
    """
    Home Controller
    :return: Renders the template for the home page with the needed variables
    """
    # clustering = 'kmeans'
    # if request.method == 'POST':
    #     clustering = request.form['clustering']
    #
    # data, tracks = load_data()
    # track_id = 2
    #
    # # get all tracks names
    # songs_names = tracks['track_title'].tolist()
    #
    # # TODO this is all wrong
    # # for k-means:
    # # k=8 -> By the inertia elbow method, but SS: ~0.15824811473835565
    # # k=8 -> SS: ~0.15824811473835565
    #
    # # for dbscan:
    # # k=5 -> By the inertia elbow method, SS: ~0.4563707621732127
    # clustering = Clustering(data, clustering, 8)
    # data = clustering.clustering_alg()
    #
    # # Initialize the metrics
    # metrics = Metrics(data)
    # # Cluster Cohesion Metric by Silhouette Score
    # metrics.cluster_cohesion()
    # metrics.davies_bouldin()
    #
    # # TODO Remove, Temporary to make a silhouette analysis and get the best number for k
    # # __get_silhouette_analysis(data)
    # # metrics.inertia()
    #
    # # data.to_csv('static/new_data.csv', sep=',')
    # print("Loading recommendation")
    # rec = __recommend(track_id, data)
    #
    # rec_tracks = tracks[tracks.index.isin(rec.index)]
    # rec_tracks_url = rec_tracks['track_url'].tolist()
    # print("Recommendation finished")
    #
    # print("Getting chosen track")
    # chosen_track = tracks[tracks.index == track_id]
    # chosen_track_url = chosen_track['track_url'].get(track_id)
    # chosen_track_mp3 = __get_track_mp3(chosen_track_url)
    # print("Chosen track done")
    #
    # print("Get Mp3 for recommended tracks")
    # # this takes a long time, but it's prettier this way
    # rec_tracks_url_mp3 = []
    # for track in rec_tracks_url:
    #     track_mp3 = __get_track_mp3(track)
    #     rec_tracks_url_mp3.append(track_mp3)
    #
    # plot_path = clustering.show()

    data, tracks = load_data()
    songs = tracks['track_title']

    return render_template('home.html', songs=tracks)
                           # chosen_track=chosen_track_mp3,
                           # rec=rec_tracks_url,
                           # rec_mp3=rec_tracks_url_mp3,
                           # songs=songs_names,
                           # plot_path=plot_path)


def process():
    data, tracks = load_data()
    track_id = int(request.json.get('song'))
    selected_algorithm = request.json.get('algorithm')
    print("track id:", track_id)
    print("algorithm:", selected_algorithm)

    # TODO this is all wrong
    # for k-means:
    # k=8 -> By the inertia elbow method, but SS: ~0.15824811473835565
    # k=8 -> SS: ~0.15824811473835565

    # for dbscan:
    # k=5 -> By the inertia elbow method, SS: ~0.4563707621732127
    clustering = Clustering(data, selected_algorithm, 8)
    data = clustering.clustering_alg()

    # # Initialize the metrics
    # metrics = Metrics(data)
    # # Cluster Cohesion Metric by Silhouette Score
    # metrics.cluster_cohesion()
    # metrics.davies_bouldin()

    # TODO Remove, Temporary to make a silhouette analysis and get the best number for k
    # __get_silhouette_analysis(data)
    # metrics.inertia()

    # data.to_csv('static/new_data.csv', sep=',')
    print("Loading recommendation")
    rec = __recommend(track_id, data)

    rec_tracks = tracks[tracks.index.isin(rec.index)]
    rec_tracks_url = rec_tracks['track_url'].tolist()
    print("Recommendation finished")

    print("Getting chosen track")
    chosen_track = tracks[tracks.index == track_id]
    chosen_track_url = chosen_track['track_url'].get(track_id)
    chosen_track_mp3 = __get_track_mp3(chosen_track_url)
    print("Chosen track done")

    print("Get Mp3 for recommended tracks")
    # this takes a long time, but it's prettier this way
    rec_tracks_url_mp3 = []
    for track in rec_tracks_url:
        track_mp3 = __get_track_mp3(track)
        rec_tracks_url_mp3.append(track_mp3)

    plot_path = clustering.show()

    return jsonify(rec_tracks_url_mp3)


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
    This way it's possible to play the mp3 on my website
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

