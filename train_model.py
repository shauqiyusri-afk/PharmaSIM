import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('sample_medicine_data.csv')  # Make sure this CSV is in the same folder

# Encode categorical features
encoders = {}
for col in ['race', 'gender', 'symptom', 'recommended_medicine']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Train model
X = df[['race', 'gender', 'age', 'symptom']]
y = df['recommended_medicine']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'medicine_model.pkl')
joblib.dump(encoders['race'], 'race_encoder.pkl')
joblib.dump(encoders['gender'], 'gender_encoder.pkl')
joblib.dump(encoders['symptom'], 'symptom_encoder.pkl')
joblib.dump(encoders['recommended_medicine'], 'recommended_medicine_encoder.pkl')

print("✅ Model and encoders saved successfully.")
