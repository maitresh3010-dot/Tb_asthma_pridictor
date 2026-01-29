import librosa
import numpy as np
import config

def extract_45_features(audio_source):
    """Safely extracts 45 MFCC features for deployment."""
    try:
        # Standardize input to 3 seconds
        y, sr = librosa.load(audio_source, sr=22050, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=45)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        # Deployment tip: Log the error but don't crash the app
        print(f"Deployment Error in extraction: {e}")
        return None