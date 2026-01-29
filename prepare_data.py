import librosa
import numpy as np
import pandas as pd
import glob
import os

def extract_45_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=45)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None

# 1. Process New Healthy Data
new_rows = []
if os.path.exists('Healthy'):
    print("🔄 Processing healthy control group...")
    files = glob.glob("Healthy/*.wav")
    for f in files:
        features = extract_45_features(f)
        if features is not None:
            # Force string keys '0' through '44'
            row = {str(i): features[i] for i in range(45)}
            row['label'] = 'NORMAL'
            new_rows.append(row)
df_new = pd.DataFrame(new_rows)

# 2. Process Original TB Data
if os.path.exists("train.csv"):
    print("📂 Loading and fixing original TB dataset...")
    df_old = pd.read_csv("train.csv")
    
    # CRITICAL: Convert all column names to strings to match df_new
    df_old.columns = [str(col) for col in df_old.columns]
    
    if 'label' not in df_old.columns:
        df_old['label'] = 'TB' 

    # 3. Merge and strictly keep only the 45 features + label
    df_master = pd.concat([df_old, df_new], ignore_index=True)
    
    # Force alignment by selecting only these specific columns
    final_cols = [str(i) for i in range(45)] + ['label']
    df_master = df_master[final_cols]
    
    # Final cleanup of any half-empty rows
    df_master = df_master.dropna()
    
    df_master.to_csv("master_dataset.csv", index=False)
    print(f"✅ Success! Master Dataset created with {len(df_master)} total samples.")
    print(df_master['label'].value_counts())
else:
    print("❌ train.csv not found! Copy it here first.")