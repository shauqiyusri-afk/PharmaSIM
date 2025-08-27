from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Import CORS
import joblib
import numpy as np

# Load model and encoders
model = joblib.load('medicine_model.pkl')
race_enc = joblib.load('race_encoder.pkl')
gender_enc = joblib.load('gender_encoder.pkl')
symptom_enc = joblib.load('symptom_encoder.pkl')
medicine_enc = joblib.load('recommended_medicine_encoder.pkl')

app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        # Extract and encode input
        race = race_enc.transform([data['race']])[0]
        gender = gender_enc.transform([data['gender']])[0]
        age = int(data['age'])
        symptom = symptom_enc.transform([data['symptom']])[0]

        input_vector = np.array([[race, gender, age, symptom]])
        prediction = model.predict(input_vector)[0]
        medicine = medicine_enc.inverse_transform([prediction])[0]

        return jsonify({'recommended_medicine': medicine})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
