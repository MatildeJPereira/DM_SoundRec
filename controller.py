from flask import render_template, request, redirect
from models.clustering import Clustering
from models.recommendation import Recommendation
import pandas as pd


def load_data():
    # load the features data taking into account the multi-level headers
    data = pd.read_csv("fma_metadata/features.csv", header=[0, 1, 2, 3])
    # join the headers into one readable level
    data.columns = ['_'.join(col).strip() for col in data.columns]
    data.columns = ['track_id' if col == 'feature_statistics_number_track_id' else col for col in data.columns]
    # Drop rows with missing values (I don't think they exist but one can never be sure)
    data = data.dropna()

    # TODO: I can select specific groups of features: now using MFCCs and Chroma CQT
    track_id = data.filter(like='track_id')
    mfcc_col = data.filter(like='mfcc')
    cqt_col = data.filter(like='chroma_cqt')
    selected_data = pd.concat([track_id, cqt_col, mfcc_col], axis=1)

    # load raw tracks data (it's easier to work with for me)
    tracks = pd.read_csv('fma_metadata/raw_tracks.csv', header=[0])
    # There's a lot of rows with missing values that we don't want to lose

    return selected_data, tracks


def home():
    data, tracks = load_data()

    clustering = Clustering(data, 'kmeans', 10)
    data = clustering.clustering_alg()
    # data.to_csv('static/new_data.csv', sep=',')
    rec = __recommend(data)

    rec_tracks = tracks[tracks['track_id'].isin(rec['track_id'])]
    rec_tracks_url = rec_tracks['track_url'].tolist()
    return render_template('home.html', rec=rec_tracks_url)


def cluster():
    data = load_data()
    clustering = Clustering(data, 'kmeans', 10)
    clustering.clustering_alg()
    plot_path = clustering.show()

    return render_template('cluster.html', plot_path=plot_path)


def __recommend(data):
    rec = Recommendation(2, data).recommend()
    return rec

