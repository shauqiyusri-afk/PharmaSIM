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
success_rate_model  = joblib.load('success_model.pkl')

race_enc    = joblib.load('race_encoder.pkl')
gender_enc  = joblib.load('gender_encoder.pkl')
symptom_enc = joblib.load('symptom_encoder.pkl')

# -----------------------------
# Known medicines + ingredients
# -----------------------------
known_meds = pd.read_csv("known_medicines.csv")

with open("ingredient_map.json", "r", encoding="utf-8") as f:
    ingredient_map = json.load(f)


def get_ingredients_for(med_name):
    entry = ingredient_map.get(med_name)
    if entry:
        return (
            set([a.strip().lower() for a in entry.get("active", [])]),
            set([i.strip().lower() for i in entry.get("inactive", [])])
        )
    return set(), set()

# ----------------- Indication ontology / hints -----------------
def _norm(s):
    return (s or "").strip().lower()


INDICATION_ALIASES = {
    "her2+": "her2+",
    "sore throat": "sore throat",
    "flu": "flu",
    "headache": "headache",
    "fever": "fever",
    "cough": "cough",
    "colon cancer": "colon cancer",
    "breast cancer": "breast cancer",
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
        "headache", "fever", "toothache", "muscle pain", "sore throat"
    },
    "allergy_upper_respiratory": {
        "flu", "cough", "allergic rhinitis", "nasal congestion", "sore throat"
    },
    "onc_colon": {"colon cancer"},
    "onc_breast": {"breast cancer", "breast cancer (her2+)"},
    "onc_lung": {"lung cancer"},
    "onc_leukemia": {"leukemia"},
    "onc_lymphoma": {"lymphoma"},
    "onc_pancreas": {"pancreatic cancer"},
    "onc_melanoma": {"melanoma"},
}

INDICATION_TO_GROUP = {}
for g, vals in INDICATION_GROUPS.items():
    for v in vals:
        INDICATION_TO_GROUP[_norm(v)] = g


def indication_group(name: str) -> str:
    return INDICATION_TO_GROUP.get(normalize_indication(name), "")


def is_oncology_group(g: str) -> bool:
    return g.startswith("onc_")


INGREDIENT_TO_HINT_GROUPS = {
    "paracetamol": {"analgesic_antipyretic"},
    "acetaminophen": {"analgesic_antipyretic"},
    "ibuprofen": {"analgesic_antipyretic"},
    "naproxen": {"analgesic_antipyretic"},
    "aspirin": {"analgesic_antipyretic"},
    "caffeine": {"analgesic_antipyretic"},
    "cetirizine": {"allergy_upper_respiratory"},
    "loratadine": {"allergy_upper_respiratory"},
    "diphenhydramine": {"allergy_upper_respiratory"},
    "pseudoephedrine": {"allergy_upper_respiratory"},
    "dextromethorphan": {"allergy_upper_respiratory"},
    "guaifenesin": {"allergy_upper_respiratory"},
    "cisplatin": set(),
    "carboplatin": set(),
    "paclitaxel": set(),
    "doxorubicin": set(),
    "cyclophosphamide": set(),
    "methotrexate": set(),
    "gemcitabine": set(),
    "imatinib": set(),
    "trastuzumab": set(),
    "bevacizumab": set(),
    "nivolumab": set(),
    "pembrolizumab": set(),
}

def score_similarity(input_symptom,
                     input_line,
                     input_ingredients_active,
                     input_ingredients_inactive,
                     input_dosage,
                     row):

    # Similarity weights
    W_SYMPTOM_EXACT        = 52.0
    W_SYMPTOM_SAME_GROUP   = 26.0
    W_LINE_EXACT           = 12.0
    W_ACTIVE_PER_MATCH     = 4.0
    W_ACTIVE_CAP           = 12.0
    W_INACTIVE_PER_MATCH   = 0.5
    W_INACTIVE_CAP         = 2.0
    W_DOSAGE_CLOSE         = 3.0

    PENALTY_CROSS_DOMAIN   = 40.0
    PENALTY_DIFF_DISEASE   = 18.0
    PENALTY_OPPOSITE_LINE  = 5.0

    W_ING_COMPAT_PER_HIT   = 3.0
    W_ING_COMPAT_CAP       = 9.0

    inp_symptom = normalize_indication(input_symptom)
    inp_line    = _norm(input_line or "general")
    inp_group   = indication_group(inp_symptom)
    inp_is_onc  = is_oncology_group(inp_group)

    row_symptom = normalize_indication(row.get('target_symptom', ''))
    row_line    = _norm(row.get('line_of_treatment', 'general'))
    row_group   = indication_group(row_symptom)
    row_is_onc  = is_oncology_group(row_group)

    row_dosage  = float(row.get('dosage_mg', 0))
    row_active, row_inactive = get_ingredients_for(row['medicine_name'])

    total = 0.0
    max_score = 0.0

    # Symptom matching
    if row_symptom == inp_symptom and inp_symptom:
        total += W_SYMPTOM_EXACT
        max_score += W_SYMPTOM_EXACT
    else:
        if inp_group and row_group and inp_group == row_group and not inp_is_onc:
            total += W_SYMPTOM_SAME_GROUP
            max_score += W_SYMPTOM_SAME_GROUP
        else:
            penalty = PENALTY_CROSS_DOMAIN if (inp_is_onc != row_is_onc) else PENALTY_DIFF_DISEASE
            total -= penalty
            max_score += max(W_SYMPTOM_EXACT, W_SYMPTOM_SAME_GROUP)

    # Line of treatment
    max_score += W_LINE_EXACT
    if inp_line and row_line and inp_line != "general" and row_line != "general":
        if row_line == inp_line:
            total += W_LINE_EXACT
        else:
            total -= PENALTY_OPPOSITE_LINE

    # Active ingredient overlap
    overlap_active = len((input_ingredients_active or set()) & (row_active or set()))
    active_score = min(W_ACTIVE_CAP, overlap_active * W_ACTIVE_PER_MATCH)
    total += active_score
    max_score += W_ACTIVE_CAP

    # Inactive ingredient overlap
    overlap_inactive = len((input_ingredients_inactive or set()) & (row_inactive or set()))
    inactive_score = min(W_INACTIVE_CAP, overlap_inactive * W_INACTIVE_PER_MATCH)
    total += inactive_score
    max_score += W_INACTIVE_CAP

    # Ingredient compatibility with indication group
    compat_hits = 0
    for ing in (input_ingredients_active or set()):
        hint_groups = INGREDIENT_TO_HINT_GROUPS.get(_norm(ing), set())
        if inp_group and inp_group in hint_groups:
            compat_hits += 1
    compat_score = min(W_ING_COMPAT_CAP, compat_hits * W_ING_COMPAT_PER_HIT)
    total += compat_score
    max_score += W_ING_COMPAT_CAP

    # Dosage proximity
    max_score += W_DOSAGE_CLOSE
    if row_dosage > 0:
        diff = abs(row_dosage - (input_dosage or 0.0))
        if diff <= 0.10 * (row_dosage + 1):
            total += W_DOSAGE_CLOSE
        elif diff <= 0.25 * (row_dosage + 1):
            total += W_DOSAGE_CLOSE * 0.5

    max_score = max(1e-6, max_score)
    percent = (total / max_score) * 100.0
    percent = max(0.0, min(100.0, percent))

    return percent, {
        "overlap_active": overlap_active,
        "overlap_inactive": overlap_inactive,
        "row_dosage": row_dosage,
        "row_line": row_line,
        "row_group": row_group,
        "input_group": inp_group
    }

def calculate_organ_function_penalties(liver_function, kidney_function, ingredients):
            """Realistic organ function adjustments based on drug metabolism"""
            penalties = []
            adjustments = {}
            
            # Convert to consistent format
            liver_status = liver_function.lower() if liver_function else "normal"
            kidney_status = kidney_function.lower() if kidney_function else "normal"
            
            # Liver impairment penalties
            if "mild" in liver_status or "moderate" in liver_status or "severe" in liver_status:
                # Drugs that are hepatically metabolized
                hepatically_cleared = ['paracetamol', 'ibuprofen', 'diazepam', 'simvastatin', 'warfarin', 'codeine']
                if any(drug in ' '.join(ingredients).lower() for drug in hepatically_cleared):
                    if "mild" in liver_status:
                        penalty = 0.15
                        adjustments['effectiveness_penalty'] = penalty
                        penalties.append("Liver impairment reduces metabolism of hepatically-cleared drugs")
                    elif "moderate" in liver_status:
                        penalty = 0.25
                        adjustments['effectiveness_penalty'] = penalty
                        penalties.append("Moderate liver impairment significantly affects drug metabolism")
                    elif "severe" in liver_status:
                        penalty = 0.40
                        adjustments['effectiveness_penalty'] = penalty
                        penalties.append("Severe liver impairment - consider alternative medications")
            
            # Kidney impairment penalties  
            if "mild" in kidney_status or "moderate" in kidney_status or "severe" in kidney_status:
                # Drugs that are renally cleared
                renally_cleared = ['metformin', 'digoxin', 'gentamicin', 'lisinopril', 'penicillin']
                if any(drug in ' '.join(ingredients).lower() for drug in renally_cleared):
                    if "mild" in kidney_status:
                        penalty = 0.10
                        adjustments['success_penalty'] = penalty
                        penalties.append("Kidney impairment affects clearance of renally-excreted drugs")
                    elif "moderate" in kidney_status:
                        penalty = 0.20
                        adjustments['success_penalty'] = penalty
                        penalties.append("Moderate kidney impairment requires dose adjustment")
                    elif "severe" in kidney_status:
                        penalty = 0.35
                        adjustments['success_penalty'] = penalty
                        penalties.append("Severe kidney impairment - avoid renally cleared drugs")
            
            return adjustments, penalties

def predict_for_all_ethnicities(base_data, ingredients, dosage_mg):
    """Run ML predictions for all ethnicities"""
    ethnicity_results = {}
    
    # Base inputs (from current prediction)
    gender_e = safe_transform(gender_enc, base_data['gender'])
    symptom_e = safe_transform(symptom_enc, base_data['symptom'])
    age = base_data['age']
    ingredient_count = len(ingredients)
    
    # Test for each ethnicity
    ethnicities = ["malay", "chinese", "indian", "indigenous"]
    
    for ethnicity in ethnicities:
        try:
            # Encode this ethnicity
            race_e = safe_transform(race_enc, ethnicity)
            
            # Create input vector
            input_vector = np.array([[
                race_e, gender_e, age, symptom_e, ingredient_count, dosage_mg
            ]])
            
            # Get ML predictions
            effectiveness = float(effectiveness_model.predict(input_vector)[0])
            side_effect = float(side_effect_model.predict(input_vector)[0])
            success_rate = float(success_rate_model.predict(input_vector)[0])
            
            # Apply bounds
            effectiveness = max(0.0, min(100.0, effectiveness))
            success_rate = max(0.0, min(100.0, success_rate))
            
            ethnicity_results[ethnicity] = {
                "effectiveness": effectiveness,
                "side_effect_risk": side_effect,
                "success_rate": success_rate
            }
            
        except Exception as e:
            # Fallback if ethnicity not in encoder
            ethnicity_results[ethnicity] = {
                "effectiveness": 75.0,  # Default fallback
                "side_effect_risk": 0.3,
                "success_rate": 70.0
            }
    
    return ethnicity_results

# -----------------------------
# User auth storage
# -----------------------------
USERS_FILE = "users.json"
SESS_FILE  = "sessions.json"

def load_json(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
        return default
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

users = load_json(USERS_FILE, {})
sessions = load_json(SESS_FILE, {})

def auth_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "").strip()
        if not token or token not in sessions:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper

# -----------------------------
# Flask app setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
CORS(app)

# ---------- Auth API ----------
@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    name  = (data.get("name")  or "").strip()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    if email in users:
        return jsonify({"error": "Email already registered"}), 400

    users[email] = {
        "name": name or email.split("@")[0],
        "password_hash": generate_password_hash(password)
    }
    save_json(USERS_FILE, users)
    return jsonify({"message": "Registered", "email": email})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    user = users.get(email)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = secrets.token_urlsafe(32)
    sessions[token] = {"email": email}
    save_json(SESS_FILE, sessions)

    return jsonify({"token": token, "user": {"email": email, "name": user["name"]}})

@app.route("/api/logout", methods=["POST"])
@auth_required
def api_logout():
    auth = request.headers.get("Authorization", "")
    token = auth.replace("Bearer ", "").strip()
    if token in sessions:
        sessions.pop(token)
        save_json(SESS_FILE, sessions)
    return jsonify({"message": "Logged out"})

# ---------- Page routes ----------
@app.route("/", methods=["GET"])
def welcome_page():
    return render_template("welcome.html")

@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")

@app.route("/app", methods=["GET"])
def app_page():
    return render_template("app.html")

# ---------- Prediction ----------
# Synonym map and normalization helpers
synonym_map = {
    "heart_disease": ["heart disease", "heart failure"],
    "liver_disease": ["liver disease", "poor liver function", "liver dysfunction"],
    "kidney_disease": ["kidney disease", "kidney dysfunction"],
    "elderly": ["elderly", "elderly (sedation risk)", "elderly (falls risk)"],
    "pregnancy": ["pregnancy"],
    "hypertension": ["hypertension", "uncontrolled hypertension"],
    "asthma": ["asthma"],
    "glaucoma": ["glaucoma"]
}

def normalize_condition(cond):
    if not cond:
        return ""
    cond = cond.lower().strip()
    cond = cond.split("(")[0].strip()
    return cond

def condition_matches(user_cond, med_risk):
    med_risk_norm = normalize_condition(med_risk)
    for frontend_cond, synonyms in synonym_map.items():
        if user_cond == frontend_cond:
            if med_risk_norm in synonyms or med_risk_norm == frontend_cond.replace("_", " "):
                return True
    return False

# --- Encode categorical inputs with fallback ---
def safe_transform(enc, value):
    try:
        return enc.transform([value])[0]
    except Exception:
        if "unknown" in enc.classes_:
            return enc.transform(["unknown"])[0]
        return 0

# -----------------------------
# Flask route: Predict
# -----------------------------
@app.route("/predict", methods=["POST"])
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

        # --- Optional extra clinical inputs (for narrative + rule-based tweaks) ---
        weight_kg_raw = data.get("weight_kg")
        try:
            weight_kg = float(weight_kg_raw) if weight_kg_raw not in (None, "",) else 0.0
        except (TypeError, ValueError):
            weight_kg = 0.0

        route = (data.get("route") or "").strip()
        try:
            treatment_duration_days = int(data.get("treatment_duration_days", 0) or 0)
        except (TypeError, ValueError):
            treatment_duration_days = 0
        dosing_frequency = (data.get("dosing_frequency") or "").strip()
        liver_function = (data.get("liver_function") or "").strip().lower()
        kidney_function = (data.get("kidney_function") or "").strip().lower()

        # --- Cancer auto-suggest ---
        cancer_type = data.get("cancer_type")
        cancer_line = data.get("cancer_line_of_treatment")
        cancer_risks = [r.lower() for r in data.get("cancer_risk_factors", [])]

        # --- Normalize conditions ---
        user_conditions = [normalize_condition(c) for c in health_conditions if c]

        # Male cannot be pregnant → strip pregnancy from conditions for male patients
        if gender.lower() == "male":
            user_conditions = [c for c in user_conditions if "pregnan" not in c]
            health_conditions = [c for c in health_conditions if "pregnan" not in c]

        # --- Dosage handling ---
        concentration = float(data.get("concentration", 0))
        dosage_mg = float(data.get("dosage_mg", 0))
        dosage_ml = float(data.get("dosage_ml", 0))
        if dosage_mg == 0 and dosage_ml > 0 and concentration > 0:
            dosage_mg = dosage_ml * concentration
        if dosage_ml == 0 and dosage_mg > 0 and concentration > 0:
            dosage_ml = dosage_mg / concentration

        # --- Get clinical safety factors ---
        genetic_risks = data.get('genetic_risks', '')
        liver_function = data.get('liver_function', 'normal')
        kidney_function = data.get('kidney_function', 'normal')

        # --- Initialize penalty variables ---
        penalty_pct = 0.0
        explanations_new = []

        # --- Ingredients ---
        tokens = [t.strip().lower() for t in ingredients_raw.split(';') if t.strip()]
        input_active = set(tokens)
        input_inactive = set()

        # Calculate REAL organ function penalties
        organ_adjustments, organ_penalties = calculate_organ_function_penalties(liver_function, kidney_function, tokens)

        # --- Encode categorical ---
        race_e = safe_transform(race_enc, race)
        gender_e = safe_transform(gender_enc, gender)
        symptom_e = safe_transform(symptom_enc, symptom)

        ingredient_count = len(tokens)
        input_vector = np.array([[race_e, gender_e, age, symptom_e, ingredient_count, dosage_mg]])

        # --- ML predictions ---
        effectiveness_raw = float(effectiveness_model.predict(input_vector)[0])
        side_effect_raw   = float(side_effect_model.predict(input_vector)[0])
        success_rate_ml   = float(success_rate_model.predict(input_vector)[0])

        # working copies for adjustments
        effectiveness = effectiveness_raw
        side_effect_val = side_effect_raw
        success_rate = success_rate_ml

        # --- STORE ETHNICITY COMPARISONS ---
        ethnicity_predictions = predict_for_all_ethnicities(
            base_data={
                'gender': gender,
                'symptom': symptom, 
                'age': age
            },
            ingredients=tokens,
            dosage_mg=dosage_mg
        )

        # Store in user session for the chart
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "").strip()
        if token in sessions:
            sessions[token]['ethnicity_predictions'] = ethnicity_predictions
            save_json(SESS_FILE, sessions)

        # Apply organ penalties to predictions
        if 'effectiveness_penalty' in organ_adjustments:
            effectiveness *= (1 - organ_adjustments['effectiveness_penalty'])
        if 'success_penalty' in organ_adjustments:
            success_rate *= (1 - organ_adjustments['success_penalty'])

        # Add organ penalties to the total penalty percentage
        if 'effectiveness_penalty' in organ_adjustments:
            penalty_pct += organ_adjustments['effectiveness_penalty']
        if 'success_penalty' in organ_adjustments:
            penalty_pct += organ_adjustments['success_penalty']

        # Add organ penalty explanations
        explanations_new.extend(organ_penalties)

        # --- Basic bounds on raw outputs ---
        # side_effect_val is a probability-like 0–1; others are %
        side_effect_val = max(0.0, min(1.0, side_effect_val))
        effectiveness = max(0.0, min(100.0, effectiveness))
        success_rate = max(0.0, min(100.0, success_rate))

        # --- Side effect label (first pass from model) ---
        if side_effect_val < 0.33:
            side_effect_label = "Low"
        elif side_effect_val < 0.66:
            side_effect_label = "Medium"
        else:
            side_effect_label = "High"

        # Helper: is this a generally "healthy, low-risk" profile?
        def is_healthy_profile():
            conds_empty = len(user_conditions) == 0
            liver_ok = (not liver_function) or ("normal" in liver_function)
            kidney_ok = (not kidney_function) or ("normal" in kidney_function)
            age_ok = 18 <= age <= 60
            # simple dose sanity for common symptomatic drugs in this prototype
            dose_ok = (dosage_mg <= 500) if dosage_mg > 0 else True
            return conds_empty and liver_ok and kidney_ok and age_ok and dose_ok

        healthy_profile = is_healthy_profile()

        # Refine side-effect label to behave more like real clinical reasoning
        if healthy_profile and side_effect_label == "High" and side_effect_val < 0.90:
            # healthy adult: only call it "High" if model is extremely sure
            side_effect_label = "Medium"
        elif (not healthy_profile) and side_effect_label == "Low" and side_effect_val > 0.25:
            # higher-risk patients: avoid over-optimistic "Low" labels
            side_effect_label = "Medium"

        # --- Age-based explanations ---
        explanations = {}
        if age > 60:
            explanations["success_rate"] = "Success rate slightly lower due to age factor."
        elif age < 18:
            explanations["success_rate"] = "Success rate adjusted for pediatric patient."
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

        # --- Hybrid clinical success rate (Option C) ---
        # OG logic: real-world success drops if side-effects are high,
        # so we combine model success with an adjusted term:
        #   success_from_tolerability = effectiveness × (1 – side_effect_val)
        success_from_tolerability = effectiveness * (1.0 - side_effect_val)

        # Blend: 70% ML success model + 30% physics-style formula
        success_rate_after_blend = 0.7 * success_rate + 0.3 * success_from_tolerability
        success_rate = success_rate_after_blend

        # Clinical sanity: overall success should not meaningfully exceed effectiveness
        if success_rate > effectiveness:
            success_rate = effectiveness

        # --- Health condition penalties ---
                # --- Health condition penalties (improved version) ---
        # Lighter weights so penalties don't destroy predictions
        health_penalty_map = {
            "liver_disease": {"note": "Liver disease may reduce drug metabolism.", "weight": 0.05},
            "kidney_disease": {"note": "Kidney impairment affects drug clearance.", "weight": 0.06},
            "asthma": {"note": "Adjusted due to asthma-related risk.", "weight": 0.03},
            "heart_disease": {"note": "Cardiovascular conditions impact drug tolerance.", "weight": 0.05},
            "hypertension": {"note": "Adjusted due to elevated blood pressure.", "weight": 0.03},
            "pregnancy": {"note": "Special safety adjustments applied for pregnancy.", "weight": 0.08},
            "glaucoma": {"note": "Certain drugs contraindicated in glaucoma.", "weight": 0.03},
            "elderly": {"note": "Elderly profile affects drug processing.", "weight": 0.04},
            "diabetes": {"note": "Metabolic changes from diabetes considered.", "weight": 0.03}
        }

        # Organ-specific structured penalties (mild/moderate/severe)
        
        penalty_pct = 0.0
        explanations_new = []
        new_drug_warning = ""

        # Apply discrete conditions
        for cond in user_conditions:
            info = health_penalty_map.get(cond)
            if info:
                penalty_pct += info["weight"]
                explanations_new.append(info["note"])

        # --- CAP THE TOTAL PENALTY ---
        penalty_pct = min(penalty_pct, 0.35)  # max 35%

        # Apply penalty to effectiveness + success rate
        if penalty_pct > 0:
            effectiveness = max(0.0, effectiveness * (1 - penalty_pct))
            success_rate = max(0.0, success_rate * (1 - penalty_pct))

            new_drug_warning = (
                f"⚠ Adjusted for patient risk factors (approx {round(penalty_pct*100)}% reduction)."
            )

            # If organ issues exist, bump side-effect label upward
            if organ_penalties:  # If there are any organ penalties
                if side_effect_label == "Low":
                    side_effect_label = "Medium"
                elif side_effect_label == "Medium":
                    # If we have moderate/severe organ issues, increase risk
                    if any("moderate" in penalty.lower() or "severe" in penalty.lower() for penalty in organ_penalties):
                        side_effect_label = "High"

            explanations["effectiveness"] = "Adjusted for metabolism/clearance factors."
            explanations["success_rate"] = "Adjusted for patient-specific risks."
            explanations["side_effects"] = "Adjusted due to physiological risk factors."

        else:
            explanations.setdefault("effectiveness", "Effectiveness remains stable.")


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
                "percent": round(percent_adjusted, 2),
                "details": details,
                "ingredients_active": list(a),
                "ingredients_inactive": list(i),
                "effectiveness": display_effectiveness,
                "success_rate": display_success_rate,
                "side_effect_risk": row.get('side_effect_risk'),
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
                    input_line = next_line
                    break

        matches_sorted = sorted(filtered_matches, key=lambda x: (x['percent'], -int(x['risky'])), reverse=True)
        top_matches = matches_sorted[:3] if matches_sorted else []

        MATCH_THRESHOLD = 55.0
        best = next((m for m in top_matches if not m['risky']), top_matches[0] if top_matches else None)
        strong_match = bool(best and best['percent'] >= MATCH_THRESHOLD and not best['risky'])

        # --- Cancer suggestions ---
        cancer_suggestions = []
        if cancer_type:
            for _, row in known_meds.iterrows():
                if str(row.get("cancer_type", "")).lower() == cancer_type.lower():
                    if not cancer_line or str(row.get("line_of_treatment", "")).lower() == cancer_line.lower():
                        cancer_suggestions.append({
                            "medicine_name": row["medicine_name"],
                            "cancer_type": row.get("cancer_type"),
                            "line_of_treatment": row.get("line_of_treatment"),
                            "effectiveness": row.get("effectiveness"),
                            "success_rate": row.get("success_rate"),
                            "risk_factors": row.get("risk_factors", "")
                        })

        # --- Predict specific side effects for NEW drugs ---
        predicted_side_effects = []

        if best and best.get("known_side_effects"):
            predicted_side_effects.extend(best["known_side_effects"].split(";"))

        if race.lower() == "malay":
            predicted_side_effects.append("Skin rash (higher risk in Malays with sulfa drugs)")
        elif race.lower() == "chinese":
            predicted_side_effects.append("Flushing or liver enzyme interaction")
        elif race.lower() == "indian":
            predicted_side_effects.append("Liver toxicity risk with paracetamol")
        elif race.lower() == "indigenous":
            predicted_side_effects.append("Hypersensitivity / dizziness")

        if dosage_mg > 500:
            predicted_side_effects.append("Nausea (dose-related)")
            predicted_side_effects.append("Dizziness (dose-related)")

        if any("liver" in c.lower() for c in user_conditions):
            predicted_side_effects.append("Liver toxicity")
        if any("pregnancy" in c.lower() for c in user_conditions):
            predicted_side_effects.append("Unsafe in pregnancy / fetal risk")
        if any("kidney" in c.lower() for c in user_conditions):
            predicted_side_effects.append("Renal impairment risk")

        predicted_side_effects = list({s.strip() for s in predicted_side_effects if s.strip()})

        # --- Ethnicity scores (mocked from prediction + matches) ---
        def get_credible_ethnicity_insights(race, ingredients, genetic_risks):
            """Real clinical insights based on pharmacogenomics"""
            insights = []
            
            # CYP2D6 Poor Metabolizers (very common in Malays)
            if race.lower() == "malay" and any('codeine' in ing.lower() for ing in ingredients):
                insights.append("50% of Malays are CYP2D6 poor metabolizers → codeine may be ineffective")
            
            # HLA-B*1502 Risk (Asian populations)
            if race.lower() in ["chinese", "malay", "indian"]:
                if genetic_risks == "hla_b1502" and any('carbamazepine' in ing.lower() for ing in ingredients):
                    insights.append("🚫 CONTRAINDICATED: High Stevens-Johnson syndrome risk in HLA-B*1502 carriers")
                elif any('carbamazepine' in ing.lower() for ing in ingredients):
                    insights.append("Screen for HLA-B*1502 before prescribing carbamazepine")
            
            # Warfarin dosing (Asian populations)
            if race.lower() in ["chinese", "malay", "indian"] and any('warfarin' in ing.lower() for ing in ingredients):
                insights.append("Asian patients typically require 30-50% lower warfarin doses")
            
            # TPMT Deficiency
            if genetic_risks == "tpmt_deficient" and any('azathioprine' in ing.lower() for ing in ingredients):
                insights.append("🚫 CONTRAINDICATED: Severe myelosuppression risk in TPMT deficient patients")
            
            # DPYD Deficiency (Fluoropyrimidines)
            if any('fluorouracil' in ing.lower() for ing in ingredients) or any('capecitabine' in ing.lower() for ing in ingredients):
                insights.append("Screen for DPYD deficiency before prescribing fluoropyrimidines")
            return insights

        # USAGE - Add this where you deleted the old code:
        genetic_risks = data.get('genetic_risks', '')
        ethnicity_insights = get_credible_ethnicity_insights(race, tokens, genetic_risks)
        # --- Clamp predictions to 0–100 range just in case ---
        effectiveness = max(0.0, min(100.0, effectiveness))
        success_rate = max(0.0, min(100.0, success_rate))

        # --- Narrative generation for frontend explanation ---
        gender_label = (gender or "").lower()
        if gender_label == "male":
            gender_phrase = "male"
        elif gender_label == "female":
            gender_phrase = "female"
        else:
            gender_phrase = gender

        race_phrase = race
        symptom_phrase = symptom

        duration_phrase = None
        if treatment_duration_days and treatment_duration_days > 0:
            # Simple humanized phrase; can be improved later
            if 28 <= treatment_duration_days <= 35:
                duration_phrase = "about 1 month"
            elif 56 <= treatment_duration_days <= 80:
                duration_phrase = "about 2–3 months"
            else:
                duration_phrase = f"about {treatment_duration_days} days"

        route_phrase = route.lower() if route else ""
        frequency_phrase = dosing_frequency

        base_sentence = f'"{drug_name}" is designed to treat {symptom_phrase} in a {age}-year-old {race_phrase} {gender_phrase} patient'
        if weight_kg and weight_kg > 0:
            base_sentence += f" weighing {weight_kg:.1f} kg"

        treatment_bits = []
        if route_phrase:
            treatment_bits.append(route_phrase)
        if frequency_phrase:
            treatment_bits.append(frequency_phrase)

        treatment_str = ""
        if treatment_bits or duration_phrase:
            inner = " ".join(treatment_bits) if treatment_bits else ""
            if duration_phrase:
                if inner:
                    inner = inner + f" for {duration_phrase}"
                else:
                    inner = f"for {duration_phrase}"
            treatment_str = f", given {inner}"

        narrative_intro = base_sentence + treatment_str + "."
        narrative_outcome = (
            f" The model predicts an estimated effectiveness of {effectiveness:.2f}%, and an overall "
            f"success rate of {success_rate:.2f}%. The success rate is computed using a hybrid method "
            f"that combines the machine-learning prediction with a clinical adjustment term "
            f"(effectiveness × tolerability), reflecting how real-world treatment success depends on both "
            f"drug efficacy and expected side-effect burden. Side-effect risk is assessed as "
            f"{side_effect_label.lower()}, influenced by dose level, patient characteristics, and any "
            f"organ-function concerns."
        )
        model_narrative = narrative_intro + narrative_outcome
                # --- Append penalty explanation if any physiological penalties applied ---
        if penalty_pct > 0:
            model_narrative += (
                f" Due to identified physiological or health-related risk factors, the system "
                f"applied an adjustment of approximately {round(penalty_pct*100)}% to both effectiveness "
                f"and success rate. This reflects clinical considerations such as reduced metabolism, "
                f"slower drug clearance, altered pharmacodynamics, or comorbidity-associated risks that "
                f"may realistically diminish treatment response or tolerability."
            )
        else:
            model_narrative += (
                " No significant physiological risk factors were detected, so no additional penalties "
                "were applied to the predicted outcomes."
            )

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
                "ingredients_provided": tokens,
                "dosage_mg": dosage_mg,
                "dosage_ml": dosage_ml,
                "concentration": concentration,
                "symptom": symptom,
                "line_of_treatment": input_line,
                "age": age,
                "race": race,
                "gender": gender,
                "health_conditions": health_conditions,
                "cancer_type": cancer_type,
                "cancer_line_of_treatment": cancer_line,
                "cancer_risk_factors": cancer_risks,
                "weight_kg": weight_kg,
                "route": route,
                "treatment_duration_days": treatment_duration_days,
                "dosing_frequency": dosing_frequency,
                "liver_function": liver_function,
                "kidney_function": kidney_function,
            },
                "clinical_insights": ethnicity_insights + organ_penalties,
                "organ_function_considerations": organ_penalties,
                "genetic_considerations": ethnicity_insights,
                "ethnicity_scores": {"note": "Replaced with clinical insights system"},
                "escalation_applied": escalation_applied,
                "model_narrative": model_narrative,
                        "debug": {
                "input_vector": input_vector.tolist(),
                "raw_predictions": {
                    "effectiveness_raw": effectiveness_raw,
                    "side_effect_raw": side_effect_raw,
                    "success_rate_ml_raw": success_rate_ml
                },
                "hybrid_success_components": {
                    "success_from_tolerability": success_from_tolerability,
                    "success_after_blend_before_penalty": success_rate_after_blend,
                    "penalty_pct": penalty_pct
                },
                "final_outputs": {
                    "effectiveness": effectiveness,
                    "success_rate": success_rate,
                    "side_effect_label": side_effect_label
                },
                "flags": {
                    "healthy_profile": healthy_profile
                },
                "matches_raw": [
                    (m["medicine_name"], m["percent"]) for m in matches
                ]
            }
        }

        if not strong_match:
            response["message"] = "No strong safe match found — showing closest alternatives."
        if escalation_applied:
            response["message"] = f"No suitable {data.get('line_of_treatment')} medicine found. Escalated to {input_line}."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ethnicity-data", methods=["GET"])
@auth_required
def ethnicity_data():
    # Get current user's ethnicity predictions
    auth = request.headers.get("Authorization", "")
    token = auth.replace("Bearer ", "").strip()
    user_session = sessions.get(token, {})
    
    ethnicity_predictions = user_session.get('ethnicity_predictions', {})
    
    # If no predictions yet, return empty data
    if not ethnicity_predictions:
        return jsonify({
            "labels": ["Malay", "Chinese", "Indian", "Indigenous"],
            "datasets": []
        })
    
    # Extract effectiveness scores for each ethnicity
    your_drug_scores = [
        ethnicity_predictions.get("malay", {}).get("effectiveness", 75),
        ethnicity_predictions.get("chinese", {}).get("effectiveness", 75),
        ethnicity_predictions.get("indian", {}).get("effectiveness", 75),
        ethnicity_predictions.get("indigenous", {}).get("effectiveness", 75)
    ]
    
    # Calculate known medicine baseline (your drug - 10% as example)
    known_medicine_scores = [score * 0.90 for score in your_drug_scores]
    
    data = {
        "labels": ["Malay", "Chinese", "Indian", "Indigenous"],
        "datasets": [
            {
                "label": "Your Drug",
                "data": your_drug_scores,
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "pointBackgroundColor": "rgb(75, 192, 192)"
            },
            {
                "label": "Known Medicine Avg",
                "data": known_medicine_scores,
                "borderColor": "rgb(255, 99, 132)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)", 
                "pointBackgroundColor": "rgb(255, 99, 132)"
            }
        ]
    }
    
    return jsonify(data)

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    for fn in ["ingredient_map.json", "known_medicines.csv", "users.json", "sessions.json"]:
        if not os.path.exists(fn):
            print(f"WARNING: missing file: {fn}")

    for folder in ["templates", "static"]:
        path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(path):
            print(f"WARNING: {folder} folder not found at {path}")

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
