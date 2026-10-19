"""
inference.py – Standalone inference module using the unified final pipeline.
Ensures identical preprocessing and model scoring between training, evaluation, and production.
"""

import os
import pandas as pd
import numpy as np
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')

if os.path.exists(os.path.join(MODELS_DIR, 'final_pipeline.joblib')):
    PIPELINE_PATH = os.path.join(MODELS_DIR, 'final_pipeline.joblib')
else:
    PIPELINE_PATH = os.path.join(FRONTEND_DIR, 'final_pipeline.joblib')


def load_pipeline():
    """Loads the complete fitted scikit-learn pipeline (preprocessor + classifier)."""
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Trained pipeline not found at {PIPELINE_PATH}. Please run train.py first.")
    return joblib.load(PIPELINE_PATH)


def load_preprocessor():
    """Loads the fitted ColumnTransformer preprocessor."""
    prep_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    if not os.path.exists(prep_path):
        prep_path = os.path.join(FRONTEND_DIR, 'preprocessor.joblib')
    return joblib.load(prep_path)


def load_model():
    """Loads the fitted classifier model."""
    model_path = os.path.join(MODELS_DIR, 'final_model.joblib')
    if not os.path.exists(model_path):
        model_path = os.path.join(FRONTEND_DIR, 'final_model.joblib')
    return joblib.load(model_path)


def compute_service_count(row: dict) -> int:
    """Computes SubscribedServiceCount across 9 services."""
    services = [
        row.get('PhoneService') == 'Yes',
        row.get('MultipleLines') == 'Yes',
        row.get('InternetService') in ['DSL', 'Fiber optic'],
        row.get('OnlineSecurity') == 'Yes',
        row.get('OnlineBackup') == 'Yes',
        row.get('DeviceProtection') == 'Yes',
        row.get('TechSupport') == 'Yes',
        row.get('StreamingTV') == 'Yes',
        row.get('StreamingMovies') == 'Yes'
    ]
    return sum(int(s) for s in services)


def predict_single(customer_data: dict, threshold: float = 0.50) -> dict:
    """
    Predicts churn for a single customer record.

    Parameters
    ----------
    customer_data : dict
        Customer attributes (matching raw features, excluding customerID and Churn).
    threshold : float
        Decision threshold for class assignment (default: 0.50).

    Returns
    -------
    dict:
        predicted_class: 'Churn' or 'No Churn'
        probability: float between 0.0 and 1.0
        risk_label: 'High Risk', 'Medium Risk', or 'Low Risk'
        threshold_used: float
    """
    pipeline = load_pipeline()
    data = customer_data.copy()

    # Automatically derive SubscribedServiceCount if not provided
    if 'SubscribedServiceCount' not in data:
        data['SubscribedServiceCount'] = compute_service_count(data)

    df_input = pd.DataFrame([data])
    prob = float(pipeline.predict_proba(df_input)[0, 1])
    predicted_class = "Churn" if prob >= threshold else "No Churn"

    if prob >= 0.70:
        risk_label = "High Risk"
    elif prob >= 0.40:
        risk_label = "Medium Risk"
    else:
        risk_label = "Low Risk"

    return {
        "predicted_class": predicted_class,
        "probability": round(prob, 4),
        "risk_label": risk_label,
        "threshold_used": threshold
    }
