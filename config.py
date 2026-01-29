# config.py
# Centralized settings for Swaas-Check V2

# Audio Processing Constants
SAMPLING_RATE = 22050
DURATION = 3
N_MFCC = 45  # Must match the 45 features in your master_dataset.csv

# File Paths
MODEL_PATH = "audio_model.pkl"
DATABASE_PATH = "swaas_check.db"

# Administrative Labels
LABELS = {
    'NORMAL': 'Healthy',
    'TB': 'Tuberculosis'
}