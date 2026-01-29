import pandas as pd

df = pd.read_csv("train.csv")

def extract_label(path):
    path = str(path).lower()
    if 'tb' in path or 'heavy' in path:
        return 'TB'
    if 'asthma' in path or 'wheeze' in path or 'shallow' in path:
        return 'ASTHMA'
    # ADD THIS: Look for files that represent healthy coughs
    if 'healthy' in path or 'normal' in path or 'v1' in path: 
        return 'NORMAL'
    return 'OTHER'

df['label'] = df['filename'].apply(extract_label)

# Keep all three categories now
df_final = df[df['label'].isin(['TB', 'ASTHMA', 'NORMAL'])]

df_final.to_csv("train_fixed.csv", index=False)
print(f"✅ New Distribution: {df_final['label'].value_counts().to_dict()}")