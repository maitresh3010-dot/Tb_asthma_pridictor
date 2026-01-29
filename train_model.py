import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the fixed master dataset
df = pd.read_csv("master_dataset.csv")

# Separate Features and Labels
X = df.drop('label', axis=1)
y = df['label']

print(f"🎯 Training on {len(X)} clean samples with {X.shape[1]} features...")

# Train with 'balanced' weights to handle the sample size difference
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X, y)

# Save the brain for the exhibition
pickle.dump(model, open("audio_model.pkl", "wb"))
print("✅ Swaas-Check V2 Model trained successfully!")