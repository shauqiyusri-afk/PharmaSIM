# PharmaSIM AI

AI-assisted drug simulation platform for predicting treatment effectiveness, success probability, and potential side-effect risk based on pharmacological and demographic parameters.

---

# Overview

PharmaSIM AI is a research and experimental project exploring how artificial intelligence and data-driven models can assist in **early-stage drug simulation and analysis**.

The system allows users to input hypothetical drug parameters such as:

- symptoms  
- ingredients  
- dosage  
- patient demographic context  

The platform then generates predicted outcomes including:

- estimated effectiveness  
- treatment success probability  
- potential side-effect risk  

The system also compares new drug inputs against a dataset of known medicines to estimate similarity and expected performance.

This project was originally developed during the **ICESCO International Hackathon** and later expanded as a personal research project exploring AI-assisted healthcare simulation.

---

# Objectives

- Simulate potential drug behavior using AI-assisted models  
- Predict treatment effectiveness and success probability  
- Estimate potential side-effect risks  
- Compare simulated drugs with known medicines  
- Provide interpretable outputs for experimental analysis  

---

# System Architecture

User Input (Symptoms / Ingredients / Dosage)  
↓  
Data Preprocessing  
↓  
Feature Encoding  
↓  
AI Prediction Models  
↓  
Similarity Analysis with Known Medicines  
↓  
Simulation Results Dashboard  

---

# Repository Structure

```
backend
Flask backend and simulation logic

models
Machine learning models and prediction logic

datasets
Example dataset used for simulation demonstration

frontend
HTML/CSS interface for user interaction

tests
Testing scripts and experimental modules
```

---

# Technologies

- Python  
- Flask  
- Random Forest (experimental prediction model)  
- Heuristic PK/PD simulation concepts  
- HTML / CSS frontend  

---

# Example Use Cases

- Early-stage drug concept simulation  
- Educational demonstrations of pharmacological modeling  
- AI-assisted treatment analysis experiments  
- Comparative drug behavior simulations  

---

# Current Status

Development and experimentation in progress.

The project currently focuses on building the **core simulation pipeline**, improving prediction interpretability, and refining similarity matching with known medicines.

---

# Future Development

- Expanded pharmacological datasets  
- Improved model explainability  
- Integration of additional machine learning models  
- More advanced PK/PD simulation logic  

---

# Backend & Code Availability

Some backend modules included in this repository are **simplified versions of the system used during development**.

The purpose of this repository is to demonstrate the **system architecture, workflow, and simulation approach** used in the PharmaSIM platform.

Certain internal components have been simplified to keep the repository easier to understand and reproduce.

---

# Dataset (.csv) Notice

Some datasets included in this repository are **reduced example datasets** intended only to demonstrate how the system works.

The complete datasets used during development are not publicly shared due to **data privacy and usage considerations**.

The backend also is the simplified version itended to show how PharmaSim works in general.
