import json

from flask import render_template, request, redirect
from models.clustering import Clustering
from models.recommendation import Recommendation
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler


def load_data():
    # # load the features data taking into account the multi-level headers
    # data = pd.read_csv("fma_metadata/features.csv", header=[0, 1, 2, 3])
    # # join the headers into one readable level
    # data.columns = ['_'.join(col).strip() for col in data.columns]
    # data.columns = ['track_id' if col == 'feature_statistics_number_track_id' else col for col in data.columns]
    # # Drop rows with missing values (I don't think they exist but one can never be sure)
    # data = data.dropna()

    # # TODO: I can select specific groups of features: now using MFCCs, Chroma CENS and Tonnetz
    # track_id = data.filter(like='track_id')
    # mfcc_col = data.filter(like='mfcc')
    # cqt_col = data.filter(like='chroma_cens')
    # tonnetz_col = data.filter(like='tonnetz')
    # selected_data = pd.concat([track_id, cqt_col, mfcc_col, tonnetz_col], axis=1)

    # load the echonest data taking into account the multi-level headers, the track_id is the index
    data = pd.read_csv("fma_metadata/raw_echonest.csv", header=[0, 1, 2], index_col=0)
    selected_data = data.loc[:, data.columns.get_level_values(level=2).isin([
        "acousticness", "danceability", "energy", "instrumentalness", "liveness", "speechiness", "tempo", "valence"])]

    selected_data = selected_data.dropna()

    # load raw tracks data, the track_id is the index
    tracks = pd.read_csv('fma_metadata/raw_tracks.csv', header=[0], index_col=0)

    # TODO Join the two tables?
    # scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)
    # this looses the index but if you turn it back to a dataframe add index=selected_data.index
    scaled_data_df = pd.DataFrame(scaled_data, index=selected_data.index, columns=selected_data.columns)

    return scaled_data_df, tracks


def home():
    data, tracks = load_data()

    clustering = Clustering(data, 'kmeans', 10)
    data = clustering.clustering_alg()
    # data.to_csv('static/new_data.csv', sep=',')
    rec = __recommend(data)

    rec_tracks = tracks[tracks.index.isin(rec.index)]
    rec_tracks_url = rec_tracks['track_url'].tolist()

    # this takes a long long time, but it's prettier this way
    rec_tracks_url_mp3 = []
    for track in rec_tracks_url:
        res = requests.get(track)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, 'html.parser')
            song_tag = soup.find('div', {'class': 'play-item'})
            if song_tag:
                song_info_str = song_tag['data-track-info']
                song_info = json.loads(song_info_str)
                rec_tracks_url_mp3.append(song_info['fileUrl'])

    print(rec_tracks_url_mp3)
    plot_path = 0
    # plot_path = clustering.show()

    return render_template('home.html', rec=rec_tracks_url, rec_mp3=rec_tracks_url_mp3, plot_path=plot_path)


def __recommend(data):
    rec = Recommendation(2, data).recommend()
    return rec

