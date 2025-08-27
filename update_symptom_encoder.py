import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your updated CSV
df = pd.read_csv("known_medicines.csv")

# Collect unique target_symptom values
symptoms = df["target_symptom"].unique().tolist()
print("All labels:", symptoms)

# Fit encoder on all current labels
symptom_enc = LabelEncoder()
symptom_enc.fit(symptoms)

# Save the updated encoder
joblib.dump(symptom_enc, "symptom_encoder.pkl")
print("Updated encoder saved with classes:", symptom_enc.classes_)
