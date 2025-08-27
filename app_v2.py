from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

# Load models
medicine_model = joblib.load('medicine_model_v2.pkl')
effectiveness_model = joblib.load('effectiveness_model.pkl')
side_effect_model = joblib.load('side_effect_model.pkl')

# Load encoders
race_enc = joblib.load('race_encoder.pkl')
gender_enc = joblib.load('gender_encoder.pkl')
symptom_enc = joblib.load('symptom_encoder.pkl')
medicine_enc = joblib.load('recommended_medicine_encoder.pkl')
side_effect_risk_enc = joblib.load('side_effect_risk_encoder.pkl')

# Load CSV to get side effect descriptions
df = pd.read_csv("extended_medicine_data.csv")
side_effect_lookup = dict(zip(df["recommended_medicine"], df["common_side_effects"]))

# Flask setup
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Encode inputs
        race = race_enc.transform([data['race']])[0]
        gender = gender_enc.transform([data['gender']])[0]
        age = int(data['age'])
        symptom = symptom_enc.transform([data['symptom']])[0]

        input_vector = np.array([[race, gender, age, symptom]])

        # Predict
        medicine_pred = medicine_model.predict(input_vector)[0]
        effectiveness_pred = effectiveness_model.predict(input_vector)[0]
        side_effect_pred = side_effect_model.predict(input_vector)[0]

        # Decode
        medicine = medicine_enc.inverse_transform([medicine_pred])[0]
        risk = side_effect_risk_enc.inverse_transform([side_effect_pred])[0]
        side_effects = side_effect_lookup.get(medicine, "Not Available")

        return jsonify({
            "recommended_medicine": medicine,
            "effectiveness_score": round(effectiveness_pred, 1),
            "side_effect_risk": risk,
            "side_effects": side_effects
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
