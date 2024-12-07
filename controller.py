from flask import render_template, request, redirect
from models.clustering import Clustering
from models.recommendation import Recommendation
import pandas as pd


def load_data():
    # load the data taking into account the multi-level headers
    data = pd.read_csv("fma_metadata/features.csv", header=[0, 1, 2, 3])
    # join the headers into one readable level
    data.columns = ['_'.join(col).strip() for col in data.columns]
    data.columns = ['track_id' if col == 'feature_statistics_number_track_id' else col for col in data.columns]
    # Drop rows with missing values (I don't think they exist but one can never be sure)
    data = data.dropna()
    return data


def home():
    data = load_data()
    clustering = Clustering(data, 'kmeans', 10)
    data = clustering.clustering_alg()
    rec = recommend(data)
    return render_template('home.html', rec=rec)


def cluster():
    data = load_data()
    clustering = Clustering(data, 'kmeans', 10)
    clustering.clustering_alg()
    plot_path = clustering.show()

    return render_template('cluster.html', plot_path=plot_path)


def recommend(data):
    print(data.columns)
    rec = Recommendation(2, data).recommend()
    return rec

