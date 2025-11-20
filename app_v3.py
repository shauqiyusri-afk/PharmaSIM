from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import os
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# -----------------------------
# Models / encoders
# -----------------------------
effectiveness_model = joblib.load('effectiveness_model.pkl')
side_effect_model   = joblib.load('side_effect_model.pkl')
success_rate_model  = joblib.load('success_rate_model.pkl')

race_enc    = joblib.load('race_encoder.pkl')
gender_enc  = joblib.load('gender_encoder.pkl')
symptom_enc = joblib.load('symptom_encoder.pkl')

# -----------------------------
# Known medicines dataset
# -----------------------------
known_meds = pd.read_csv('known_medicines_dataset.csv')

# -----------------------------
# Ingredients mapping
# -----------------------------
def load_json(filename, default):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return default

ingredient_map = load_json('ingredient_map.json', {})
active_ingredient_map = load_json('active_ingredient_map.json', {})

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)
CORS(app)

# Simple in-memory user store (for demo)
USERS_DB = {}

# Token store
TOKENS = set()

def create_token():
    return secrets.token_hex(16)

# -----------------------------
# Auth decorators
# -----------------------------
def auth_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Missing authorization header"}), 401

        token = auth_header.replace("Bearer ", "").strip()
        if not token or token not in TOKENS:
            return jsonify({"error": "Invalid or expired token"}), 401

        return f(*args, **kwargs)
    return wrapper

# -----------------------------
# Helper functions
# -----------------------------
def safe_transform(encoder, value, default=0):
    try:
        return encoder.transform([value])[0]
    except Exception:
        return default

def get_ingredients_for(med_name):
    entry = ingredient_map.get(med_name)
    if isinstance(entry, dict):
        a = entry.get("active", [])
        i = entry.get("inactive", [])
        if isinstance(a, str):
            a = [x.strip().lower() for x in a.split(';') if x.strip()]
        if isinstance(i, str):
            i = [x.strip().lower() for x in i.split(';') if x.strip()]
        return set(a), set(i)
    elif isinstance(entry, list):
        # treat entire list as "active"
        a = [x.strip().lower() for x in entry if isinstance(x, str) and x.strip()]
        return set(a), set()
    elif isinstance(entry, str):
        tokens = [x.strip().lower() for x in entry.split(';') if x.strip()]
        return set(tokens), set()
    else:
        # fallback empty sets
        return set(), set()

def score_similarity(symptom, line, input_active, input_inactive, dosage_mg, row):
    symptom_score = 0
    line_score    = 0
    ingredient_score = 0
    dosage_score = 0

    row_symptom = str(row.get('target_symptom', '')).lower()
    if row_symptom == symptom.lower():
        symptom_score = 40
    elif symptom.lower() in row_symptom or row_symptom in symptom.lower():
        symptom_score = 25

    row_line = str(row.get('line_of_treatment', 'general')).lower()
    if row_line == line:
        line_score = 20
    elif ("first" in line and "second" in row_line) or ("second" in line and "first" in row_line):
        line_score = 10
    else:
        line_score = 5

    # ingredients
    a, i = get_ingredients_for(row['medicine_name'])
    if input_active and a:
        intersect = input_active.intersection(a)
        if intersect:
            ingredient_score = 30 * (len(intersect) / max(1, len(a)))
    # We do not penalize for inactive mismatch; just a small boost if any match
    if input_inactive and i:
        if input_inactive.intersection(i):
            ingredient_score += 5

    # dosage
    row_dose = float(row.get('dosage_mg', 0))
    if dosage_mg and row_dose:
        ratio = dosage_mg / row_dose
        if 0.8 <= ratio <= 1.2:
            dosage_score = 10
        elif 0.5 <= ratio < 0.8 or 1.2 < ratio <= 1.5:
            dosage_score = 5
        else:
            dosage_score = 2

    total = symptom_score + line_score + ingredient_score + dosage_score
    total = min(100, total)
    return total, {
        "symptom_score": symptom_score,
        "line_score": line_score,
        "ingredient_score": ingredient_score,
        "dosage_score": dosage_score,
    }

# -----------------------------
# Cancer indication helpers
# -----------------------------
def _norm(s: str) -> str:
    return str(s).strip().lower()

CANCER_ALIASES = {
    "breast cancer": "breast cancer",
    "breast ca": "breast cancer",
    "ca breast": "breast cancer",
    "her2+ breast cancer": "breast cancer (her2+)",
    "her2 positive breast cancer": "breast cancer (her2+)",
    "her2+": "breast cancer (her2+)",
    "lung ca": "lung cancer",
    "ca lung": "lung cancer",
    "nsclc": "lung cancer",
    "cml": "leukemia",
    "aml": "leukemia",
    "all": "leukemia",
    "hodgkin lymphoma": "lymphoma",
    "non-hodgkin lymphoma": "lymphoma",
    "pancreatic ca": "pancreatic cancer",
    "melanoma": "melanoma",
}

def normalize_cancer_type(name: str) -> str:
    n = _norm(name)
    return CANCER_ALIASES.get(n, n)

INDICATION_ALIASES = {
    "fever": "fever",
    "pyrexia": "fever",
    "headache": "headache",
    "migraine": "headache",
    "muscle pain": "muscle pain",
    "myalgia": "muscle pain",
    "tooth pain": "toothache",
    "toothache": "toothache",
    "sore throat": "sore throat",
    "flu": "flu",
    "influenza": "flu",
    "covid": "flu",
    "covid-19": "flu",
    "breast cancer": "breast cancer",
    "breast ca": "breast cancer",
    "ca breast": "breast cancer",
    "breast cancer (her2+)": "breast cancer (her2+)",
    "lung cancer": "lung cancer",
    "leukemia": "leukemia",
    "lymphoma": "lymphoma",
    "pancreatic cancer": "pancreatic cancer",
    "melanoma": "melanoma",
}

def normalize_indication(name: str) -> str:
    n = _norm(name)
    return INDICATION_ALIASES.get(n, n)

INDICATION_GROUPS = {
    "analgesic_antipyretic": {
        "headache", "fever", "toothache", "muscle pain", "sore throat", "flu"
    },
    "oncology": {
        "breast cancer", "breast cancer (her2+)", "lung cancer", "leukemia",
        "lymphoma", "pancreatic cancer", "melanoma"
    },
}

def condition_matches(user_cond: str, risk_factor: str) -> bool:
    """
    Very simple helper to say if a given user condition
    (e.g. 'liver_disease') should be considered matched
    to a row-level risk factor string.
    """
    uc = user_cond.lower()
    rf = risk_factor.lower()
    if uc == "liver_disease":
        return "liver" in rf or "hepat" in rf
    if uc == "kidney_disease":
        return "kidney" in rf or "renal" in rf
    if uc == "asthma":
        return "asthma" in rf or "bronchospasm" in rf
    if uc == "heart_disease":
        return "heart" in rf or "cardio" in rf
    if uc == "hypertension":
        return "hypertension" in rf or "high blood pressure" in rf
    if uc == "pregnancy":
        return "pregnan" in rf or "fetal" in rf
    if uc == "glaucoma":
        return "glaucoma" in rf or "intraocular pressure" in rf
    if uc == "elderly":
        return "elderly" in rf or "geriatric" in rf
    if uc == "diabetes":
        return "diabetes" in rf or "hyperglycemia" in rf
    return uc in rf

def normalize_condition(cond: str) -> str:
    c = cond.lower().strip()
    mapping = {
        "liver disease": "liver_disease",
        "kidney disease": "kidney_disease",
        "heart disease": "heart_disease",
        "high blood pressure": "hypertension",
        "pregnant": "pregnancy",
        "pregnancy": "pregnancy",
        "glaucoma": "glaucoma",
        "elderly": "elderly",
        "diabetic": "diabetes",
        "diabetes": "diabetes"
    }
    return mapping.get(c, c)

# -----------------------------
# Ethnicity scoring
# -----------------------------
def generate_ethnicity_scores(effectiveness, success_rate, side_effect_val, race):
    """
    Generate a simple ethnicity performance profile.
    Values are heuristic but anchored on your base predictions.
    """
    base_eff = effectiveness / 100.0
    base_succ = success_rate / 100.0

    # Start around baseline; different race slightly up/down
    scores_new = {
        "Malay": {
            "effectiveness": round(60 + 25 * base_eff, 1),
            "success": round(58 + 25 * base_succ, 1),
            "safety": round(70 - 30 * side_effect_val, 1)
        },
        "Chinese": {
            "effectiveness": round(62 + 23 * base_eff, 1),
            "success": round(60 + 23 * base_succ, 1),
            "safety": round(68 - 28 * side_effect_val, 1)
        },
        "Indian": {
            "effectiveness": round(58 + 27 * base_eff, 1),
            "success": round(56 + 27 * base_succ, 1),
            "safety": round(66 - 32 * side_effect_val, 1)
        },
        "Indigenous": {
            "effectiveness": round(55 + 30 * base_eff, 1),
            "success": round(54 + 30 * base_succ, 1),
            "safety": round(64 - 35 * side_effect_val, 1)
        }
    }

    # Known medicine baseline: slightly less tailored, more "average"
    scores_known = {
        "Malay": {
            "effectiveness": scores_new["Malay"]["effectiveness"] - 3,
            "success": scores_new["Malay"]["success"] - 3,
            "safety": scores_new["Malay"]["safety"]
        },
        "Chinese": {
            "effectiveness": scores_new["Chinese"]["effectiveness"] - 2,
            "success": scores_new["Chinese"]["success"] - 2,
            "safety": scores_new["Chinese"]["safety"]
        },
        "Indian": {
            "effectiveness": scores_new["Indian"]["effectiveness"] - 2,
            "success": scores_new["Indian"]["success"] - 2,
            "safety": scores_new["Indian"]["safety"] - 1
        },
        "Indigenous": {
            "effectiveness": scores_new["Indigenous"]["effectiveness"] - 1,
            "success": scores_new["Indigenous"]["success"] - 1,
            "safety": scores_new["Indigenous"]["safety"] - 1
        }
    }

    return {
        "new_drug": scores_new,
        "known_medicine": scores_known
    }

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/app')
def app_page():
    return render_template('app.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

# -----------------------------
# Auth endpoints
# -----------------------------
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"error": "Name, email and password are required."}), 400
    if email in USERS_DB:
        return jsonify({"error": "Email already registered."}), 400

    USERS_DB[email] = {
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password)
    }
    return jsonify({"message": "User registered successfully."}), 201

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json() or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    user = USERS_DB.get(email)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password."}), 401

    token = create_token()
    TOKENS.add(token)
    return jsonify({
        "message": "Login successful.",
        "token": token,
        "user": {"name": user["name"], "email": user["email"]}
    })

# -----------------------------
# Predict endpoint
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # --- User inputs ---
        drug_name = data.get('drug_name', 'NewDrug')
        race = data['race']
        gender = data['gender']
        age = int(data['age'])
        symptom = data['target_symptom']
        ingredients_raw = data['ingredients']
        health_conditions = [c.lower() for c in data.get("health_conditions", [])]
        input_line = str(data.get('line_of_treatment', 'general')).lower().strip()

        # Optional extra clinical context (safe if frontend does not send these yet)
        route = str(data.get("route_of_administration", "") or data.get("route", "")).lower().strip()
        treatment_duration = str(data.get("treatment_duration", "")).lower().strip()
        weight_band = str(data.get("weight_band", "")).lower().strip()

        # --- Cancer auto-suggest ---
        cancer_type = data.get("cancer_type")
        cancer_line = data.get("cancer_line_of_treatment")
        cancer_risks = [r.lower() for r in data.get("cancer_risk_factors", [])]

        # --- Normalize conditions ---
        user_conditions = [normalize_condition(c) for c in health_conditions if c]

        # --- Logical constraint: male profile cannot be pregnant ---
        if gender.lower() == "male" and "pregnancy" in user_conditions:
            # Strip pregnancy from both normalized + raw lists so model never treats male as pregnant
            user_conditions = [c for c in user_conditions if c != "pregnancy"]
            health_conditions = [
                c for c in health_conditions
                if normalize_condition(c) != "pregnancy"
            ]

        # --- Dosage handling ---
        concentration = float(data.get("concentration", 0))
        dosage_mg = float(data.get("dosage_mg", 0))
        dosage_ml = float(data.get("dosage_ml", 0))
        if dosage_mg == 0 and dosage_ml > 0 and concentration > 0:
            dosage_mg = dosage_ml * concentration
        if dosage_ml == 0 and dosage_mg > 0 and concentration > 0:
            dosage_ml = dosage_mg / concentration

        # --- Ingredients ---
        tokens = [t.strip().lower() for t in ingredients_raw.split(';') if t.strip()]
        input_active = set(tokens)
        input_inactive = set()

        # --- Encode categorical ---
        race_e = safe_transform(race_enc, race)
        gender_e = safe_transform(gender_enc, gender)
        symptom_e = safe_transform(symptom_enc, symptom)

        ingredient_count = len(tokens)

        # --- Input vector ---
        input_vector = np.array([[race_e, gender_e, age, symptom_e,
                                  ingredient_count, dosage_mg]])

        # --- Base model predictions ---
        effectiveness = float(effectiveness_model.predict(input_vector)[0])
        side_effect_val = float(side_effect_model.predict(input_vector)[0])
        success_rate = float(success_rate_model.predict(input_vector)[0])

        # --- Side effect label (initial, may be updated later) ---
        if side_effect_val < 0.33:
            side_effect_label = "Low"
        elif side_effect_val < 0.66:
            side_effect_label = "Medium"
        else:
            side_effect_label = "High"

        # --- Age-based explanations ---
        explanations = {}
        if age > 60:
            explanations["success_rate"] = "Success rate slightly lower due to age factor."
        elif age < 18:
            explanations["success_rate"] = "Adjusted for pediatric patient."
        else:
            explanations["success_rate"] = "Success rate remains stable."

        # --- Dosage explanation ---
        if dosage_mg > 0 and dosage_mg > 500:
            explanations["side_effects"] = "Higher dosage increases side effect risk."
        elif side_effect_val > 0.66:
            explanations["side_effects"] = "High predicted side effect risk."
        elif side_effect_val > 0.33:
            explanations["side_effects"] = "Moderate predicted side effect risk."
        else:
            explanations["side_effects"] = "Side effect risk is low."

        # --- Health condition penalties ---
        health_penalty_map = {
            "liver_disease": {"note": "Effectiveness and success adjusted due to liver disease.", "weight": 0.08},
            "kidney_disease": {"note": "Adjusted for kidney disease.", "weight": 0.10},
            "asthma": {"note": "Higher risk predicted due to asthma.", "weight": 0.07},
            "heart_disease": {"note": "Adjusted due to cardiovascular risk.", "weight": 0.09},
            "hypertension": {"note": "Reduced success rate due to hypertension risk.", "weight": 0.05},
            "pregnancy": {"note": "Special caution due to pregnancy safety.", "weight": 0.12},
            "glaucoma": {"note": "Warning: contraindicated risk for glaucoma.", "weight": 0.06},
            "elderly": {"note": "Adjusted for elderly patient.", "weight": 0.05},
            "diabetes": {"note": "Effectiveness slightly reduced due to diabetes.", "weight": 0.04}
        }

        penalty_pct = 0
        explanations_new = []
        new_drug_warning = ""
        for cond in user_conditions:
            info = health_penalty_map.get(cond)
            if info:
                penalty_pct += info["weight"]
                explanations_new.append(info["note"])

        if penalty_pct > 0:
            effectiveness = max(0, effectiveness * (1 - penalty_pct))
            success_rate = max(0, success_rate * (1 - penalty_pct))
            new_drug_warning = f"⚠ Predicted effectiveness/success reduced by {round(penalty_pct*100)}% due to health conditions."
            explanations["effectiveness"] = "Effectiveness adjusted due to health conditions."
            explanations["success_rate"] = "Success rate adjusted due to health conditions."
            explanations["side_effects"] = "Side effect risk may be higher due to selected conditions."
        else:
            explanations.setdefault("effectiveness", "Effectiveness remains stable.")

        # --- Route / duration / weight adjustments (rule-based) ---
        route_risk_note = ""
        duration_risk_note = ""
        weight_risk_note = ""

        # Route: IV / parenteral plus serious comorbidities -> more toxicity
        high_risk_routes = {"iv", "intravenous"}
        serious_conditions = {"heart_disease", "kidney_disease", "liver_disease"}
        if route in high_risk_routes and any(c in user_conditions for c in serious_conditions):
            side_effect_val = min(0.99, side_effect_val + 0.08)
            route_risk_note = "IV route in a patient with cardiovascular / renal / hepatic risk — higher systemic toxicity expected."

        # Treatment duration: chronic use -> cumulative toxicity; short acute course may improve success
        long_term_labels = {"chronic", "long-term", "long term", "maintenance"}
        short_course_labels = {"short", "short-course", "acute"}
        if treatment_duration in long_term_labels:
            side_effect_val = min(0.99, side_effect_val + 0.05)
            duration_risk_note = "Long-term / maintenance use — cumulative side effects may increase."
        elif treatment_duration in short_course_labels:
            success_rate = min(100.0, success_rate + 3)

        # Weight band: approximate exposure; high fixed dose in low body weight
        try:
            dose_mg = float(dosage_mg)
        except Exception:
            dose_mg = 0.0

        high_exposure = False
        if dose_mg > 0:
            if weight_band in {"<50kg", "<50", "underweight"}:
                high_exposure = dose_mg >= 500
            elif weight_band in {">80kg", ">80", "high"}:
                high_exposure = dose_mg >= 1000

        if high_exposure:
            side_effect_val = min(0.99, side_effect_val + 0.05)
            success_rate = min(100.0, success_rate + 2)
            weight_risk_note = "High fixed dose relative to patient weight — higher exposure per kilogram."

        for note in (route_risk_note, duration_risk_note, weight_risk_note):
            if note:
                explanations_new.append(note)

        # Re-evaluate side effect label after these adjustments
        if side_effect_val < 0.33:
            side_effect_label = "Low"
        elif side_effect_val < 0.66:
            side_effect_label = "Medium"
        else:
            side_effect_label = "High"

        # Align qualitative explanation with updated side effect label if not already more specific
        if side_effect_label == "High":
            explanations.setdefault(
                "side_effects",
                "High predicted side effect risk given route, duration, dosage and patient profile."
            )
        elif side_effect_label == "Medium":
            explanations.setdefault(
                "side_effects",
                "Moderate predicted side effect risk given route, duration, dosage and patient profile."
            )
        else:
            explanations.setdefault(
                "side_effects",
                "Side effect risk remains low for this regimen and patient profile."
            )

        # --- Similarity matching ---
        matches = []
        for _, row in known_meds.iterrows():
            percent, details = score_similarity(symptom, input_line, input_active, input_inactive, dosage_mg, row)
            a, i = get_ingredients_for(row['medicine_name'])

            row_penalty = 0
            risk_reasons = []
            row_explanations = []
            if 'risk_factors' in row and isinstance(row['risk_factors'], str):
                row_risks = [r.strip().lower() for r in row['risk_factors'].split(';')]
                for risk in row_risks:
                    for user_cond in user_conditions:
                        if condition_matches(user_cond, risk):
                            row_penalty += 5
                            risk_reasons.append(f"Risk for condition: {user_cond}")
                            row_explanations.append(f"Reduced by 5% due to {user_cond} (from {row['medicine_name']})")

            risky = row_penalty > 0
            percent_adjusted = max(0, percent - row_penalty)
            display_effectiveness = max(0, float(row.get('effectiveness', 0)) - row_penalty)
            display_success_rate = max(0, float(row.get('success_rate', 0)) - row_penalty)
            warning_note = "⚠ Reduced effectiveness/success due to health condition risk." if risky else ""

            matches.append({
                "medicine_name": row['medicine_name'],
                "target_symptom": row.get('target_symptom', ''),
                "line_of_treatment": str(row.get('line_of_treatment', 'general')).lower(),
                "dosage_mg": row.get('dosage_mg', ''),
                "concentration": row.get('concentration', ''),
                "similarity_percent": round(percent_adjusted, 1),
                "raw_similarity_percent": round(percent, 1),
                "similarity_breakdown": details,
                "effectiveness": display_effectiveness,
                "success_rate": display_success_rate,
                "side_effect_risk": row.get('side_effect_risk', ''),
                "known_side_effects": row.get('known_side_effects'),
                "risk_factors": row.get('risk_factors', ''),
                "risky": risky,
                "risk_reasons": risk_reasons,
                "note": warning_note,
                "explanations": row_explanations
            })

        # --- Line escalation ---
        line_order = ["first-line", "second-line", "third-line", "general"]
        escalation_applied = False
        def filter_by_line(line):
            return [m for m in matches if m["line_of_treatment"] == line]

        filtered_matches = filter_by_line(input_line)
        if not filtered_matches:
            for next_line in line_order:
                if next_line == input_line:
                    continue
                filtered_matches = filter_by_line(next_line)
                if filtered_matches:
                    escalation_applied = True
                    break

        if filtered_matches:
            filtered_matches.sort(key=lambda x: x["similarity_percent"], reverse=True)
        top_matches = filtered_matches[:3] if filtered_matches else []

        strong_match = False
        best = None
        if top_matches:
            best = top_matches[0]
            if best["similarity_percent"] >= 60 and not best["risky"]:
                strong_match = True

        # --- Cancer suggestions (very simple triage) ---
        cancer_suggestions = []
        if cancer_type and cancer_line:
            cancer_type_norm = normalize_cancer_type(cancer_type)
            cancer_line_norm = str(cancer_line).lower().strip()
            for _, row in known_meds.iterrows():
                row_ind = normalize_indication(row.get('target_symptom', ''))
                row_line = str(row.get('line_of_treatment', 'general')).lower()
                if row_ind == cancer_type_norm and row_line == cancer_line_norm:
                    cancer_suggestions.append({
                        "medicine_name": row['medicine_name'],
                        "line_of_treatment": row_line,
                        "dosage_mg": row.get('dosage_mg', ''),
                        "notes": row.get('notes', '')
                    })
            if not cancer_suggestions:
                cancer_suggestions.append({
                    "medicine_name": "No direct matched regimen found.",
                    "line_of_treatment": cancer_line_norm,
                    "dosage_mg": "",
                    "notes": "Dataset may not contain this exact indication+line combo yet."
                })

        # --- Specific side effect suggestions (very simple heuristic) ---
        predicted_side_effects = []
        # Very simplified; in reality, this comes from pharmacology DB
        if "liver_disease" in user_conditions:
            predicted_side_effects.append("Hepatotoxicity & liver enzyme elevation")
        if "kidney_disease" in user_conditions:
            predicted_side_effects.append("Renal impairment / worsening kidney function")
        if "heart_disease" in user_conditions:
            predicted_side_effects.append("Arrhythmia / cardiovascular events")
        if "pregnancy" in user_conditions:
            predicted_side_effects.append("Potential fetal toxicity / teratogenicity")
        if not predicted_side_effects and side_effect_label != "Low":
            predicted_side_effects.append("General systemic side effects (e.g. nausea, fatigue)")

        # --- Ethnicity scores ---
        ethnicity_scores = generate_ethnicity_scores(effectiveness, success_rate, side_effect_val, race)

        # --- AI recovery curves (dummy shaped curves based on success & risk) ---
        def build_curve(base_success, side_val):
            # Create 6-month curve; faster rise with higher success, flatter with more side effects
            months = [0, 1, 2, 3, 4, 5, 6]
            curve = []
            for m in months:
                fraction = m / 6.0
                val = base_success * (0.4 * fraction + 0.6 * (1 - side_val) * (1 - (1 - fraction) ** 2))
                curve.append(round(val, 1))
            return curve

        ai_curve_new = build_curve(success_rate, side_effect_val)
        ai_curve_best = build_curve(best["success_rate"], side_effect_val) if best else build_curve(success_rate * 0.9, side_effect_val)

        # --- Response JSON ---
        response = {
            "predicted_effectiveness": round(effectiveness, 2),
            "predicted_side_effect_risk": side_effect_label,
            "predicted_specific_side_effects": ";".join(predicted_side_effects),
            "specific_side_effects": ";".join(predicted_side_effects),
            "predicted_success_rate": round(success_rate, 2),
            "new_drug_note": new_drug_warning,
            "new_drug_explanations": explanations_new,
            "explanations": explanations,
            "top_matches": top_matches,
            "best_match": best if strong_match else None,
            "strong_match": strong_match,
            "cancer_suggestions": cancer_suggestions,
            "input_summary": {
                "drug_name": drug_name,
                "race": race,
                "gender": gender,
                "age": age,
                "target_symptom": symptom,
                "health_conditions": health_conditions,
                "line_of_treatment": input_line,
                "ingredients": ingredients_raw,
                "dosage_mg": dosage_mg,
                "dosage_ml": dosage_ml,
                "concentration": concentration,
                "route_of_administration": route,
                "treatment_duration": treatment_duration,
                "weight_band": weight_band,
            },
            "ethnicity_scores": ethnicity_scores,
            "ai_curve_new_drug": ai_curve_new,
            "ai_curve_best_match": ai_curve_best,
            "escalation_applied": escalation_applied
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
