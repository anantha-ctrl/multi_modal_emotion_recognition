import librosa
import numpy as np

def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfccs.T  # Shape: (Time, 40)
