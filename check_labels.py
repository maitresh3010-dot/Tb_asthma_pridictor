import pandas as pd 
import os
if os.path.exists("train.csv"):
    df = pd.read_csv("train.csv")
    print(" ---- dataset checks ----")
    print(f"total rows: {len(df)}")
    print(f"Total rows: {len(df)}")
    print("Unique labels found in your CSV:")
    for label in df['label'].unique():
        print(f" - {label}")
else:
    print("train.csv not found in this folder!")
