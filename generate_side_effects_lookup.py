import joblib

# Define a custom dictionary for specific side effects
side_effects_lookup = {
    "Malay_Headache_Paracetamol;Caffeine": "Dry Mouth, Dizziness",
    "Chinese_Cough_Dextromethorphan;Guaifenesin": "Skin Rash, Dry Throat",
    "Indian_Fever_Paracetamol;Ibuprofen": "Upset Stomach, Sweating",
    "Indigenous_Flu_Chlorpheniramine;Pseudoephedrine": "Sleepiness, Irritation",
    "Malay_Sore Throat_Clarithromycin": "Dry Throat, Nausea",
    # 🔽 Add more entries if you want...
}

# Save to a file
joblib.dump(side_effects_lookup, "side_effects_lookup.pkl")

print("side_effects_lookup.pkl has been created successfully.")
