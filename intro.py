import librosa
from librosa import feature
import matplotlib.pyplot as plt
import numpy as np


def intro_example():
    return librosa.load('audio/PinkPanther30.wav', duration=30)


def chroma_comparison():
    # Load the audio as a waveform "y".
    # Store the sampling rate as "sr"
    y, sr = intro_example()

    # This is a beat tracker
    # tempo is the tempo of the song
    # beat_frames is where the beats happened
    # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # print(tempo)

    # This passes frames to timestamps
    # beat_times = librosa.frames_to_time(beat_frames,sr=sr)
    # print(beat_times)

    chroma_stft = feature.chroma_stft(y=y, sr=sr)
    chroma_cqt = feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = feature.chroma_cens(y=y, sr=sr)

    fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True)
    librosa.display.specshow(chroma_stft,
                             y_axis='chroma', x_axis='time', ax=ax[0])
    ax[0].set(title='chroma_stft')
    ax[0].label_outer()

    img = librosa.display.specshow(chroma_cqt,
                                   y_axis='chroma', x_axis='time', ax=ax[1])
    ax[1].set(title='chroma_cqt')
    ax[1].label_outer()

    librosa.display.specshow(chroma_cens,
                             y_axis='chroma', x_axis='time', ax=ax[2])
    ax[2].set(title='chroma_cens')

    fig.colorbar(img, ax=ax)

    plt.show()


def mfcc_visualization():
    return "we"