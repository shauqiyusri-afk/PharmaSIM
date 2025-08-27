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

    max_score += W_LINE_EXACT
    if inp_line and row_line and inp_line != "general" and row_line != "general":
        if row_line == inp_line:
            total += W_LINE_EXACT
        else:
            total -= PENALTY_OPPOSITE_LINE

    overlap_active = len((input_ingredients_active or set()) & (row_active or set()))
    active_score = min(W_ACTIVE_CAP, overlap_active * W_ACTIVE_PER_MATCH)
    total += active_score
    max_score += W_ACTIVE_CAP

    overlap_inactive = len((input_ingredients_inactive or set()) & (row_inactive or set()))
    inactive_score = min(W_INACTIVE_CAP, overlap_inactive * W_INACTIVE_PER_MATCH)
    total += inactive_score
    max_score += W_INACTIVE_CAP

    compat_hits = 0
    for ing in (input_ingredients_active or set()):
        hint_groups = INGREDIENT_TO_HINT_GROUPS.get(_norm(ing), set())
        if inp_group and inp_group in hint_groups:
            compat_hits += 1
    compat_score = min(W_ING_COMPAT_CAP, compat_hits * W_ING_COMPAT_PER_HIT)
    total += compat_score
    max_score += W_ING_COMPAT_CAP

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

        # --- Cancer auto-suggest ---
        cancer_type = data.get("cancer_type")
        cancer_line = data.get("cancer_line_of_treatment")
        cancer_risks = [r.lower() for r in data.get("cancer_risk_factors", [])]

        # --- Normalize conditions ---
        user_conditions = [normalize_condition(c) for c in health_conditions if c]

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
        input_vector = np.array([[race_e, gender_e, age, symptom_e, ingredient_count, dosage_mg]])

        # --- ML predictions ---
        effectiveness = float(effectiveness_model.predict(input_vector)[0])
        side_effect_val = float(side_effect_model.predict(input_vector)[0])
        success_rate = float(success_rate_model.predict(input_vector)[0])

        # --- Side effect label ---
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

        # From best match (if available)
        if best and best.get("known_side_effects"):
            predicted_side_effects.extend(best["known_side_effects"].split(";"))

        # Race modifiers
        if race.lower() == "malay":
            predicted_side_effects.append("Skin rash (higher risk in Malays with sulfa drugs)")
        elif race.lower() == "chinese":
            predicted_side_effects.append("Flushing or liver enzyme interaction")
        elif race.lower() == "indian":
            predicted_side_effects.append("Liver toxicity risk with paracetamol")
        elif race.lower() == "indigenous":
            predicted_side_effects.append("Hypersensitivity / dizziness")

        # Dosage effects
        if dosage_mg > 500:
            predicted_side_effects.append("Nausea (dose-related)")
            predicted_side_effects.append("Dizziness (dose-related)")

        # Health condition effects
        # Health condition effects (case-insensitive, keyword-based)
        if any("liver" in c.lower() for c in user_conditions):
            predicted_side_effects.append("Liver toxicity")

        if any("pregnancy" in c.lower() for c in user_conditions):
            predicted_side_effects.append("Unsafe in pregnancy / fetal risk")

        if any("kidney" in c.lower() for c in user_conditions):
            predicted_side_effects.append("Renal impairment risk")

        # Clean up duplicates
        predicted_side_effects = list({s.strip() for s in predicted_side_effects if s.strip()})

        # --- Response JSON ---
        response = {
            "predicted_effectiveness": round(effectiveness, 2),
            "predicted_side_effect_risk": side_effect_label,
            "predicted_specific_side_effects": ";".join(predicted_side_effects),  # ✅ new style
            "specific_side_effects": ";".join(predicted_side_effects),            # ✅ alias for frontend compatibility
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
                "cancer_risk_factors": cancer_risks
            },
            "escalation_applied": escalation_applied,
            "debug": {
                "input_vector": input_vector.tolist(),
                "raw_predictions": [effectiveness, side_effect_val, success_rate],
                "matches_raw": [(m["medicine_name"], m["percent"]) for m in matches]
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
def ethnicity_data():
    # 🔥 For now, return mock numbers (later we can link to CSV or DB)
    data = {
        "labels": ["Effectiveness", "Success Rate", "Side Effect Risk"],
        "datasets": [
            {
                "label": "Malay",
                "data": [70, 68, 75]
            },
            {
                "label": "Chinese",
                "data": [72, 69, 78]
            },
            {
                "label": "Indian",
                "data": [68, 65, 80]
            },
            {
                "label": "Indigenous",
                "data": [73, 71, 77]
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