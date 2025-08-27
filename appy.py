from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    # Dummy predictions
    predicted_effectiveness = random.randint(50, 95)
    predicted_success_rate = random.randint(50, 95)
    predicted_side_effect_risk = random.choice(["Low", "Medium", "High"])

    # Dummy best match
    best_match = {
        "medicine_name": "Paracetamol",
        "effectiveness": 85,
        "success_rate": 90,
        "side_effect_risk": "Low"
    }

    # Optional ethnicity scores
    ethnicity_scores = {
        "Malay": 72, 
        "Chinese": 65, 
        "Indian": 81, 
        "Indigenous": 58
    }

    # Optional top matches
    top_matches = [
        {
            "medicine_name": "Paracetamol",
            "percent": 95,
            "known_side_effects": "Nausea; Dizziness"
        }
    ]

    response = {
        "predicted_effectiveness": predicted_effectiveness,
        "predicted_success_rate": predicted_success_rate,
        "predicted_side_effect_risk": predicted_side_effect_risk,
        "best_match": best_match,
        "ethnicity_scores": ethnicity_scores,
        "top_matches": top_matches
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
