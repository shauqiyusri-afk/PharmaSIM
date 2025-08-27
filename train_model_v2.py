import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("extended_medicine_data.csv")

# Encode categorical features
race_enc = LabelEncoder()
gender_enc = LabelEncoder()
symptom_enc = LabelEncoder()
medicine_enc = LabelEncoder()
side_effect_risk_enc = LabelEncoder()

df['race'] = race_enc.fit_transform(df['race'])
df['gender'] = gender_enc.fit_transform(df['gender'])
df['symptom'] = symptom_enc.fit_transform(df['symptom'])
df['recommended_medicine'] = medicine_enc.fit_transform(df['recommended_medicine'])
df['side_effect_risk'] = side_effect_risk_enc.fit_transform(df['side_effect_risk'])

X = df[['race', 'gender', 'age', 'symptom']]
y_medicine = df['recommended_medicine']
y_effectiveness = df['effectiveness_score']
y_side_effect_risk = df['side_effect_risk']

# Train models
medicine_model = RandomForestClassifier()
medicine_model.fit(X, y_medicine)

effectiveness_model = RandomForestRegressor()
effectiveness_model.fit(X, y_effectiveness)

side_effect_model = RandomForestClassifier()
side_effect_model.fit(X, y_side_effect_risk)

# Save models
joblib.dump(medicine_model, 'medicine_model_v2.pkl')
joblib.dump(effectiveness_model, 'effectiveness_model.pkl')
joblib.dump(side_effect_model, 'side_effect_model.pkl')

# Save encoders
joblib.dump(race_enc, 'race_encoder.pkl')
joblib.dump(gender_enc, 'gender_encoder.pkl')
joblib.dump(symptom_enc, 'symptom_encoder.pkl')
joblib.dump(medicine_enc, 'recommended_medicine_encoder.pkl')
joblib.dump(side_effect_risk_enc, 'side_effect_risk_encoder.pkl')

print("✅ Models and encoders saved successfully.")
