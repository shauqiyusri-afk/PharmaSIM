import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("new_drug_simulation_data.csv")

# Process ingredients count
df['ingredients_count'] = df['ingredients'].apply(lambda x: len(str(x).split(';')))
df['dosage_mg'] = df['dosage_mg'].astype(int)

# Encode categorical features
race_enc = LabelEncoder()
gender_enc = LabelEncoder()
symptom_enc = LabelEncoder()

df['race_encoded'] = race_enc.fit_transform(df['race'])
df['gender_encoded'] = gender_enc.fit_transform(df['gender'])
df['symptom_encoded'] = symptom_enc.fit_transform(df['target_symptom'])

# Save encoders
joblib.dump(race_enc, 'race_encoder.pkl')
joblib.dump(gender_enc, 'gender_encoder.pkl')
joblib.dump(symptom_enc, 'symptom_encoder.pkl')

# Final input features
X = df[['race_encoded', 'gender_encoded', 'age', 'symptom_encoded', 'ingredients_count', 'dosage_mg']]

# Train models
effectiveness_model = RandomForestRegressor()
effectiveness_model.fit(X, df['known_effectiveness'])
joblib.dump(effectiveness_model, 'effectiveness_model.pkl')

side_effect_model = RandomForestRegressor()
side_effect_model.fit(X, df['side_effect_risk'].astype('category').cat.codes)
joblib.dump(side_effect_model, 'side_effect_model.pkl')

success_model = RandomForestRegressor()
success_model.fit(X, df['success_rate'])
joblib.dump(success_model, 'success_model.pkl')

df['key'] = df['race'] + "_" + df['target_symptom'] + "_" + df['ingredients']
side_effects_dict = dict(zip(df['key'], df['specific_side_effects']))
joblib.dump(side_effects_dict, 'side_effects_lookup.pkl')

print("Models and encoders saved successfully!")
