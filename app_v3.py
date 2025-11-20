from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import os
import secrets
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# =========================
#  Paths / constants
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

USERS_FILE = os.path.join(BASE_DIR, "users.json")
SESS_FILE  = os.path.join(BASE_DIR, "sessions.json")

# =========================
#  Helper JSON I/O
# =========================
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

users    = load_json(USERS_FILE, {})
sessions = load_json(SESS_FILE, {})

# =========================
#  Models / encoders
# =========================
effectiveness_model = joblib.load(os.path.join(BASE_DIR, "effectiveness_model.pkl"))
side_effect_model   = joblib.load(os.path.join(BASE_DIR, "side_effect_model.pkl"))
success_rate_model  = joblib.load(os.path.join(BASE_DIR, "success_model.pkl"))

race_enc    = joblib.load(os.path.join(BASE_DIR, "race_encoder.pkl"))
gender_enc  = joblib.load(os.path.join(BASE_DIR, "gender_encoder.pkl"))
symptom_enc = joblib.load(os.path.join(BASE_DIR, "symptom_encoder.pkl"))

# =========================
#  Known medicines + ingredient map
# =========================
known_meds = pd.read_csv(os.path.join(BASE_DIR, "known_medicines.csv"))

with open(os.path.join(BASE_DIR, "ingredient_map.json"), "r", encoding="utf-8") as f:
    ingredient_map = json.load(f)

def get_ingredients_for(med_name: str):
    entry = ingredient_map.get(med_name)
    if entry:
        act = {a.strip().lower() for a in entry.get("active", [])}
        inc = {i.strip().lower() for i in entry.get("inactive", [])}
        return act, inc
    return set(), set()

# =========================
#  Normalisation helpers
# =========================
def _norm(s: str) -> str:
    return (s or "").strip().lower()

# simple condition normalisation
def normalize_condition(name: str) -> str:
    s = _norm(name)
    mapping = {
        "htn": "hypertension",
        "high blood pressure": "hypertension",
        "dm": "diabetes",
        "kidney": "kidney_disease",
        "renal": "kidney_disease",
        "ckd": "kidney_disease",
        "liver": "liver_disease",
        "hepatic": "liver_disease",
        "pregnant": "pregnancy",
        "pregnancy": "pregnancy",
        "heart": "heart_disease",
        "cvd": "heart_disease",
        "elderly": "elderly",
        "old": "elderly",
        "glaucoma": "glaucoma",
        "asthma": "asthma",
    }
    return mapping.get(s, s)

def condition_matches(user_cond: str, risk_str: str) -> bool:
    """Rough matching between user condition & risk_factors text in CSV."""
    u = normalize_condition(user_cond)
    r = _norm(risk_str)
    if not u or not r:
        return False
    if u in r:
        return True
    if u == "pregnancy" and "pregnant" in r:
        return True
    if u == "kidney_disease" and ("renal" in r or "kidney" in r):
        return True
    if u == "liver_disease" and ("hepatic" in r or "liver" in r):
        return True
    return False

def safe_transform(encoder, value: str) -> int:
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        return 0

# =========================
#  Indication ontology (simplified)
# =========================
INDICATION_GROUPS = {
    "respiratory": {
        "flu", "cough", "sore throat", "cold", "upper respiratory infection"
    },
    "general_pain_fever": {
        "headache", "migraine", "fever", "body ache", "toothache"
    },
    "onc_breast": {"breast cancer", "breast cancer (her2+)"},
    "onc_colon": {"colon cancer"},
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

def normalize_indication(name: str) -> str:
    return _norm(name)

def indication_group(name: str) -> str:
    return INDICATION_TO_GROUP.get(normalize_indication(name), "")

def is_oncology_group(g: str) -> bool:
    return g.startswith("onc_")

# =========================
#  Similarity scoring
# =========================
def score_similarity(
    input_symptom,
    input_line,
    input_ingredients_active,
    input_ingredients_inactive,
    input_dosage,
    row,
):
    W_SYMPTOM_EXACT        = 52.0
    W_SYMPTOM_SAME_GROUP   = 26.0
    W_LINE_EXACT           = 12.0
    W_ACTIVE_PER_MATCH     = 4.0
    W_ACTIVE_CAP           = 12.0
    W_INACTIVE_PER_MATCH   = 0.5
    W_INACTIVE_CAP         = 2.0
    W_DOSAGE_CLOSE         = 3.0
    W_ING_COMPAT_PER_HIT   = 1.0
    W_ING_COMPAT_CAP       = 5.0

    PENALTY_DIFF_DISEASE   = 18.0
    PENALTY_CROSS_DOMAIN   = 35.0
    PENALTY_OPPOSITE_LINE  = 12.0

    inp_symptom  = normalize_indication(input_symptom)
    row_symptom  = normalize_indication(row.get("target_symptom", ""))
    inp_group    = indication_group(inp_symptom)
    row_group    = indication_group(row_symptom)

    inp_is_onc   = is_oncology_group(inp_group)
    row_is_onc   = is_oncology_group(row_group)

    inp_line = (input_line or "").strip().lower()
    row_line = str(row.get("line_of_treatment", "general")).strip().lower()

    row_dosage = float(row.get("dosage_mg", 0) or 0)
    row_active, row_inactive = get_ingredients_for(row["medicine_name"])

    total = 0.0
    max_score = 0.0

    # symptom
    if row_symptom == inp_symptom and inp_symptom:
        total += W_SYMPTOM_EXACT
        max_score += W_SYMPTOM_EXACT
    else:
        if inp_group and row_group and inp_group == row_group and not inp_is_onc:
            total += W_SYMPTOM_SAME_GROUP
            max_score += W_SYMPTOM_SAME_GROUP
        else:
            penalty = (
                PENALTY_CROSS_DOMAIN if inp_is_onc != row_is_onc
                else PENALTY_DIFF_DISEASE
            )
            total -= penalty
            max_score += max(W_SYMPTOM_EXACT, W_SYMPTOM_SAME_GROUP)

    # line_of_treatment
    max_score += W_LINE_EXACT
    if inp_line and row_line and inp_line != "general" and row_line != "general":
        if row_line == inp_line:
            total += W_LINE_EXACT
        else:
            total -= PENALTY_OPPOSITE_LINE

    # ingredient overlap
    overlap_active = len((input_ingredients_active or set()) & (row_active or set()))
    active_score = min(W_ACTIVE_CAP, overlap_active * W_ACTIVE_PER_MATCH)
    total += active_score
    max_score += W_ACTIVE_CAP

    overlap_inactive = len((input_ingredients_inactive or set()) & (row_inactive or set()))
    inactive_score = min(W_INACTIVE_CAP, overlap_inactive * W_INACTIVE_PER_MATCH)
    total += inactive_score
    max_score += W_INACTIVE_CAP

    # “compatibility” hint – if we have any ingredients at all
    compat_hits = 0
    for ing in (input_ingredients_active or set()):
        if ing in row_active:
            compat_hits += 1
    compat_score = min(W_ING_COMPAT_CAP, compat_hits * W_ING_COMPAT_PER_HIT)
    total += compat_score
    max_score += W_ING_COMPAT_CAP

    # dosage closeness
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

    details = {
        "overlap_active": overlap_active,
        "overlap_inactive": overlap_inactive,
        "row_dosage": row_dosage,
        "row_line": row_line,
        "row_symptom": row_symptom,
        "row_group": row_group,
    }
    return percent, details

# =========================
#  Auth helpers
# =========================
def auth_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "").strip()
        if not token or token not in sessions:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper

# =========================
#  Flask app
# =========================
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
CORS(app)

# ---------- Auth API ----------
@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    name  = (data.get("name") or "").strip()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    if email in users:
        return jsonify({"error": "Email already registered"}), 400

    users[email] = {
        "name": name or email.split("@")[0],
        "password_hash": generate_password_hash(password),
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

    token = secrets.token_hex(16)
    sessions[token] = {"email": email}
    save_json(SESS_FILE, sessions)
    return jsonify({"token": token, "name": user["name"]})

@app.route("/api/me", methods=["GET"])
@auth_required
def api_me():
    auth = request.headers.get("Authorization", "")
    token = auth.replace("Bearer ", "").strip()
    sess = sessions.get(token)
    return jsonify({"email": sess["email"]})

# ---------- HTML ROUTES ----------
@app.route("/")
def home():
    """Root: redirect to the welcome page."""
    return redirect(url_for("welcome"))

@app.route("/welcome")
def welcome():
    return render_template("welcome.html")

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/app")
def app_page():
    """PharmaSim main UI page."""
    return render_template("app.html")

# =========================
#  Narrative builder
# =========================
def build_narrative(input_summary, effectiveness, success_rate, side_effect_label):
    s = input_summary or {}
    # patient description
    age = s.get("age")
    race = (s.get("race") or "").lower()
    gender = (s.get("gender") or "").lower()
    weight = s.get("weight_kg")

    age_part = f"{age}-year-old " if age else ""
    race_part = f"{race} " if race else ""
    gender_part = f"{gender} " if gender else "patient"
    weight_part = f" weighing {weight} kg" if weight else ""

    patient_part = f"{age_part}{race_part}{gender_part}patient{weight_part}".strip()

    # route, frequency, duration
    route = (s.get("route") or "oral").lower()
    if route == "iv":
        route_txt = "intravenously"
    elif route in ("po", "oral"):
        route_txt = "orally"
    else:
        route_txt = route

    freq_map = {
        "od": "once daily",
        "bd": "twice daily",
        "tds": "three times daily",
        "tid": "three times daily",
        "qid": "four times daily",
        "q8h": "every 8 hours",
        "q6h": "every 6 hours",
        "weekly": "once weekly",
        "monthly": "once monthly",
    }
    freq_raw = (s.get("dosing_frequency") or "").lower()
    freq_txt = freq_map.get(freq_raw, freq_raw)

    duration_txt = ""
    dur_days = s.get("treatment_duration_days")
    if isinstance(dur_days, int) and dur_days > 0:
        if dur_days % 30 == 0 and dur_days >= 30:
            months = dur_days // 30
            duration_txt = f" for about {months} month{'s' if months > 1 else ''}"
        else:
            duration_txt = f" for {dur_days} days"

    dosing_part = ""
    if freq_txt:
        dosing_part = f", given {route_txt} {freq_txt}{duration_txt}"
    else:
        dosing_part = f", given {route_txt}{duration_txt}"

    drug_name = s.get("drug_name") or "this candidate drug"
    symptom = s.get("symptom") or "the selected symptom"

    sentence1 = (
        f"Drug {drug_name} is designed to treat {symptom} in a {patient_part}{dosing_part}."
    )

    eff_pct = int(round(effectiveness * 100))
    succ_pct = int(round(success_rate * 100))

    sentence2 = (
        f" The model predicts approximately {eff_pct}% effectiveness and "
        f"{succ_pct}% overall success rate, with {side_effect_label.lower()} side-effect risk."
    )

    return sentence1 + sentence2

# =========================
#  Core prediction logic
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # --- Basic user inputs ---
        drug_name = data.get("drug_name", "NewDrug")
        race      = data.get("race", "")
        gender    = data.get("gender", "")
        age       = int(data.get("age", 0) or 0)
        symptom   = data.get("target_symptom", "")
        ingredients_raw = data.get("ingredients", "")
        health_conditions_raw = [c.lower() for c in data.get("health_conditions", [])]
        input_line = str(data.get("line_of_treatment", "general")).lower().strip()

        # --- Helper converters ---
        def _to_float(val, default=0.0):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        def _to_int(val, default=0):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        # --- Extended clinical inputs ---
        weight_kg = _to_float(data.get("weight_kg"), 0.0)
        if weight_kg <= 0:
            weight_kg = None

        route = (data.get("route") or "oral").strip().lower()
        dosing_frequency      = (data.get("dosing_frequency") or "").strip().lower()
        treatment_duration_days = _to_int(data.get("treatment_duration_days"), 0)

        liver_status  = max(0, min(3, _to_int(data.get("liver_status"), 0)))
        kidney_status = max(0, min(3, _to_int(data.get("kidney_status"), 0)))

        pregnancy_status = (data.get("pregnancy_status") or "none").strip().lower()
        # backend safety: male can never be pregnant
        if gender.lower() != "female":
            pregnancy_status = "none"

        # --- Cancer auto-suggest fields (optional) ---
        cancer_type = data.get("cancer_type")
        cancer_line = data.get("cancer_line_of_treatment")
        cancer_risks = [r.lower() for r in data.get("cancer_risk_factors", [])]

        # --- Normalise health conditions ---
        user_conditions = [normalize_condition(c) for c in health_conditions_raw if c]

        if pregnancy_status != "none" and "pregnancy" not in user_conditions:
            user_conditions.append("pregnancy")

        # --- Dosage handling ---
        concentration = _to_float(data.get("concentration"), 0.0)
        dosage_mg = _to_float(data.get("dosage_mg"), 0.0)
        dosage_ml = _to_float(data.get("dosage_ml"), 0.0)

        if dosage_mg == 0 and dosage_ml > 0 and concentration > 0:
            dosage_mg = dosage_ml * concentration
        if dosage_ml == 0 and dosage_mg > 0 and concentration > 0:
            dosage_ml = dosage_mg / concentration

        # --- Ingredients ---
        tokens = [t.strip().lower() for t in str(ingredients_raw).split(";") if t.strip()]
        input_active   = set(tokens)
        input_inactive = set()

        # --- Encode categorical ---
        race_e    = safe_transform(race_enc, race)
        gender_e  = safe_transform(gender_enc, gender)
        symptom_e = safe_transform(symptom_enc, symptom)

        ingredient_count = len(tokens)
        input_vector = np.array([[race_e, gender_e, age, symptom_e, ingredient_count, dosage_mg]])

        # --- Raw ML predictions (0–1 space) ---
        eff_val  = float(effectiveness_model.predict(input_vector)[0])
        se_val   = float(side_effect_model.predict(input_vector)[0])
        succ_val = float(success_rate_model.predict(input_vector)[0])

        effectiveness = eff_val
        side_effect_val = se_val
        success_rate = succ_val

        # --- Side effect label ---
        if side_effect_val < 0.33:
            side_effect_label = "Low"
        elif side_effect_val < 0.66:
            side_effect_label = "Medium"
        else:
            side_effect_label = "High"

        explanations = {}

        # Basic explanation scaffold
        if age > 60:
            explanations["success_rate"] = "Success rate slightly lower due to age factor."
        elif age < 18:
            explanations["success_rate"] = "Success rate adjusted for pediatric patient."
        else:
            explanations["success_rate"] = "Success rate remains stable."

        if dosage_mg > 0 and dosage_mg > 500:
            explanations["side_effects"] = "Higher dosage increases side effect risk."
        elif side_effect_val > 0.66:
            explanations["side_effects"] = "High predicted side effect risk."
        elif side_effect_val > 0.33:
            explanations["side_effects"] = "Moderate predicted side effect risk."
        else:
            explanations["side_effects"] = "Side effect risk is low."

        # ================================
        #  New clinical heuristics
        # ================================

        # 1) Dose per kg (if weight known)
        if weight_kg:
            dose_per_kg = dosage_mg / weight_kg if dosage_mg > 0 else 0.0
            if dose_per_kg > 10:  # arbitrary proto-threshold
                side_effect_val = min(1.0, side_effect_val + 0.08)
                success_rate = max(0.0, success_rate * 0.95)
                explanations["dose_per_kg"] = (
                    f"Dose per kg ({dose_per_kg:.1f} mg/kg) is relatively high; "
                    "slightly increasing side effect risk and reducing success rate."
                )
            elif 0 < dose_per_kg < 1:
                effectiveness = max(0.0, effectiveness * 0.96)
                explanations["dose_per_kg"] = (
                    f"Dose per kg ({dose_per_kg:.1f} mg/kg) is on the lower side; "
                    "effectiveness may be slightly reduced."
                )

        # 2) Route of administration
        if route in ("iv", "intravenous"):
            side_effect_val = min(1.0, side_effect_val + 0.05)
            explanations["route"] = (
                "Intravenous route gives higher systemic exposure; "
                "side effect risk is slightly increased."
            )
        elif route in ("oral", "po"):
            explanations.setdefault(
                "route",
                "Oral route is assumed; systemic exposure is moderate for most drugs."
            )

        # 3) Organ function severity
        organ_penalty_factor = 1.0
        if liver_status > 0:
            organ_penalty_factor -= 0.03 * liver_status
            explanations["liver_status"] = "Predictions adjusted due to reduced liver function."
        if kidney_status > 0:
            organ_penalty_factor -= 0.03 * kidney_status
            explanations["kidney_status"] = "Predictions adjusted due to reduced kidney function."

        if organ_penalty_factor < 1.0:
            effectiveness = max(0.0, effectiveness * organ_penalty_factor)
            success_rate  = max(0.0, success_rate * organ_penalty_factor)

        # 4) Pregnancy status (for females only)
        if pregnancy_status != "none":
            explanations["pregnancy"] = (
                "Special caution applied because patient is pregnant; "
                "effectiveness and success are slightly reduced and safety concerns highlighted."
            )
            effectiveness = max(0.0, effectiveness * 0.9)
            success_rate  = max(0.0, success_rate * 0.9)

        # 5) Treatment duration heuristics
        if treatment_duration_days > 0:
            if treatment_duration_days > 90:
                side_effect_val = min(1.0, side_effect_val + 0.05)
                explanations["duration"] = (
                    "Long treatment duration may increase cumulative side effect risk."
                )
            elif treatment_duration_days < 14:
                explanations.setdefault(
                    "duration",
                    "Short treatment duration; outcomes mainly reflect acute response."
                )

        # 6) Dosing frequency heuristics (simple)
        freq = dosing_frequency.lower()
        if freq in ("bd", "twice daily", "q12h"):
            effectiveness = min(1.0, effectiveness * 1.03)
            side_effect_val = min(1.0, side_effect_val + 0.02)
            explanations["dosing_frequency"] = (
                "Twice-daily dosing slightly boosts effect but also increases side-effect risk."
            )
        elif freq in ("tds", "tid", "three times daily", "q8h", "qid", "four times daily"):
            effectiveness = min(1.0, effectiveness * 1.05)
            side_effect_val = min(1.0, side_effect_val + 0.04)
            explanations["dosing_frequency"] = (
                "High-frequency dosing increases both effectiveness and side-effect risk."
            )

        # Recompute side_effect_label after adjustments
        if side_effect_val < 0.33:
            side_effect_label = "Low"
        elif side_effect_val < 0.66:
            side_effect_label = "Medium"
        else:
            side_effect_label = "High"

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
            "diabetes": {"note": "Effectiveness slightly reduced due to diabetes.", "weight": 0.04},
        }

        penalty_pct = 0.0
        explanations_new = []
        new_drug_warning = ""

        for cond in user_conditions:
            info = health_penalty_map.get(cond)
            if info:
                penalty_pct += info["weight"]
                explanations_new.append(info["note"])

        if penalty_pct > 0:
            penalty_pct = min(penalty_pct, 0.7)
            effectiveness = max(0.0, effectiveness * (1 - penalty_pct))
            success_rate  = max(0.0, success_rate * (1 - penalty_pct))
            new_drug_warning = (
                f"⚠ Predicted effectiveness/success reduced by "
                f"{round(penalty_pct * 100)}% due to health conditions."
            )
            explanations["effectiveness"] = "Effectiveness adjusted due to health conditions."
            explanations["success_rate"]  = "Success rate adjusted due to health conditions."
            explanations["side_effects"]  = "Side effect risk may be higher due to selected conditions."
        else:
            explanations.setdefault("effectiveness", "Effectiveness remains stable.")

        # --- Similarity matching against known medicines ---
        matches = []
        for _, row in known_meds.iterrows():
            percent, details = score_similarity(
                symptom,
                input_line,
                input_active,
                input_inactive,
                dosage_mg,
                row,
            )
            a, i = get_ingredients_for(row["medicine_name"])

            row_penalty = 0
            risk_reasons = []
            row_explanations = []

            if "risk_factors" in row and isinstance(row["risk_factors"], str):
                row_risks = [r.strip().lower() for r in row["risk_factors"].split(";")]
                for risk in row_risks:
                    for user_cond in user_conditions:
                        if condition_matches(user_cond, risk):
                            row_penalty += 5
                            risk_reasons.append(f"Risk for condition: {user_cond}")
                            row_explanations.append(
                                f"Reduced by 5% due to {user_cond} (from {row['medicine_name']})"
                            )

            risky = row_penalty > 0
            percent_adjusted = max(0, percent - row_penalty)
            display_effectiveness = max(0, float(row.get("effectiveness", 0)) - row_penalty)
            display_success_rate  = max(0, float(row.get("success_rate", 0)) - row_penalty)
            warning_note = "⚠ Reduced effectiveness/success due to health condition risk." if risky else ""

            matches.append(
                {
                    "medicine_name": row["medicine_name"],
                    "target_symptom": row.get("target_symptom", ""),
                    "line_of_treatment": str(row.get("line_of_treatment", "general")).lower(),
                    "dosage_mg": row.get("dosage_mg", ""),
                    "percent": round(percent_adjusted, 2),
                    "details": details,
                    "ingredients_active": list(a),
                    "ingredients_inactive": list(i),
                    "effectiveness": display_effectiveness,
                    "success_rate": display_success_rate,
                    "side_effect_risk": row.get("side_effect_risk"),
                    "known_side_effects": row.get("known_side_effects"),
                    "risk_factors": row.get("risk_factors", ""),
                    "risky": risky,
                    "risk_reasons": risk_reasons,
                    "note": warning_note,
                    "explanations": row_explanations,
                }
            )

        # --- Line escalation ---
        line_order = ["first-line", "second-line", "third-line", "general"]

        def filter_by_line(line_name):
            return [m for m in matches if m["line_of_treatment"] == line_name]

        filtered_matches = filter_by_line(input_line)
        escalation_applied = False
        original_line = input_line

        if not filtered_matches:
            for next_line in line_order:
                if next_line == input_line:
                    continue
                filtered_matches = filter_by_line(next_line)
                if filtered_matches:
                    escalation_applied = True
                    input_line = next_line
                    break

        matches_sorted = sorted(
            filtered_matches,
            key=lambda x: (x["percent"], -int(x["risky"])),
            reverse=True,
        )
        top_matches = matches_sorted[:3] if matches_sorted else []

        MATCH_THRESHOLD = 55.0
        best = next(
            (m for m in top_matches if not m["risky"]),
            top_matches[0] if top_matches else None,
        )
        strong_match = bool(best and best["percent"] >= MATCH_THRESHOLD and not best["risky"])

        # --- Cancer suggestions (optional usage) ---
        cancer_suggestions = []
        if cancer_type:
            for _, row in known_meds.iterrows():
                if str(row.get("cancer_type", "")).lower() == str(cancer_type).lower():
                    if not cancer_line or str(row.get("line_of_treatment", "")).lower() == str(cancer_line).lower():
                        cancer_suggestions.append(
                            {
                                "medicine_name": row["medicine_name"],
                                "cancer_type": row.get("cancer_type"),
                                "line_of_treatment": row.get("line_of_treatment"),
                                "effectiveness": row.get("effectiveness"),
                                "success_rate": row.get("success_rate"),
                                "risk_factors": row.get("risk_factors", ""),
                            }
                        )

        # --- Predict specific side effects for NEW drugs ---
        predicted_side_effects = []

        if best and best.get("known_side_effects"):
            predicted_side_effects.extend(str(best["known_side_effects"]).split(";"))

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

        if any("liver" in c for c in user_conditions):
            predicted_side_effects.append("Liver toxicity")
        if any("pregnancy" in c for c in user_conditions):
            predicted_side_effects.append("Unsafe in pregnancy / fetal risk")
        if any("kidney" in c for c in user_conditions):
            predicted_side_effects.append("Renal impairment risk")

        predicted_side_effects = list({s.strip() for s in predicted_side_effects if s.strip()})

        # --- Ethnicity scores (heuristic) ---
        ethnicities = ["Malay", "Chinese", "Indian", "Indigenous"]
        ethnicity_scores = {"new_drug": {}, "known_medicine": {}}
        for eth in ethnicities:
            ethnicity_scores["new_drug"][eth] = round(
                effectiveness * 100 * (0.95 + 0.05 * np.random.rand()),
                1,
            )
            known_scores = [m["effectiveness"] for m in top_matches if m]
            ethnicity_scores["known_medicine"][eth] = (
                round(
                    (np.mean(known_scores) if known_scores else 0) * (0.95 + 0.05 * np.random.rand()),
                    1,
                )
                if known_scores
                else 0
            )

        # --- Build input summary for narrative ---
        input_summary = {
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
            "health_conditions": health_conditions_raw,
            "cancer_type": cancer_type,
            "cancer_line_of_treatment": cancer_line,
            "cancer_risk_factors": cancer_risks,
            "weight_kg": weight_kg,
            "route": route,
            "dosing_frequency": dosing_frequency,
            "treatment_duration_days": treatment_duration_days,
            "liver_status": liver_status,
            "kidney_status": kidney_status,
            "pregnancy_status": pregnancy_status,
        }

        narrative_summary = build_narrative(input_summary, effectiveness, success_rate, side_effect_label)

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
            "input_summary": input_summary,
            "ethnicity_scores": ethnicity_scores,
            "escalation_applied": escalation_applied,
            "narrative_summary": narrative_summary,
            "debug": {
                "input_vector": input_vector.tolist(),
                "raw_predictions": [eff_val, se_val, succ_val],
                "matches_raw": [(m["medicine_name"], m["percent"]) for m in matches],
                "original_line": original_line,
            },
        }

        if not strong_match:
            response["message"] = "No strong safe match found — showing closest alternatives."
        if escalation_applied:
            response["message"] = (
                response.get("message", "") + " " +
                f"No suitable {original_line} medicine found. Escalated to {input_line}."
            ).strip()

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional stub for future ethnicity-data endpoint
@app.route("/ethnicity-data", methods=["GET"])
def ethnicity_data():
    return jsonify({"status": "ok"})

# =========================
#  Entrypoint
# =========================
if __name__ == "__main__":
    for fn in ["ingredient_map.json", "known_medicines.csv", "users.json", "sessions.json"]:
        p = os.path.join(BASE_DIR, fn)
        if not os.path.exists(p):
            print(f"WARNING: missing file: {p}")

    for folder in ["templates", "static"]:
        path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(path):
            print(f"WARNING: {folder} folder not found at {path}")

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
