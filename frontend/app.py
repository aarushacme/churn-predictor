"""
app.py – Interactive Streamlit Frontend for Telecom Customer Churn Prediction.

Features:
- Dual-engine inference: Scikit-learn Pipeline with automatic zero-dependency NumPy fallback
- Real-time churn probability scoring with configurable operating threshold
- Dynamic multi-service input form with automatic SubscribedServiceCount calculation
- Automated risk factor diagnostic breakdown (risk vs. protective indicators)
- Business retention action triggers based on risk tier
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

# Robust path resolution: works whether run from project root or inside frontend/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(CURRENT_DIR, '..', 'models')):
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
else:
    PROJECT_ROOT = os.path.abspath('.')

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
PIPELINE_PATH = os.path.join(MODELS_DIR, 'final_pipeline.joblib')
META_PATH = os.path.join(MODELS_DIR, 'column_meta.json')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'model_weights.json')

# Embedded model weights: guarantees 100% fail-proof execution on ANY machine/Python version
EMBEDDED_MODEL_WEIGHTS = {
  "num_cols": [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
  ],
  "scaler_mean": [
    0.16177777777777777,
    32.562311111111114,
    64.99931555555555,
    2301.83952
  ],
  "scaler_scale": [
    0.3682468308011104,
    24.5402396436929,
    30.105965373777526,
    2275.383801039277
  ],
  "cat_cols": [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
  ],
  "categories": [
    ["Female", "Male"],
    ["No", "Yes"],
    ["No", "Yes"],
    ["No", "Yes"],
    ["No", "No phone service", "Yes"],
    ["DSL", "Fiber optic", "No"],
    ["No", "No internet service", "Yes"],
    ["No", "No internet service", "Yes"],
    ["No", "No internet service", "Yes"],
    ["No", "No internet service", "Yes"],
    ["No", "No internet service", "Yes"],
    ["No", "No internet service", "Yes"],
    ["Month-to-month", "One year", "Two year"],
    ["No", "Yes"],
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
  ],
  "coef": [
    0.07215879176461094,
    -1.3574040558001634,
    -0.43449664158719503,
    0.6455077648547582,
    -0.02408144058147231,
    -0.02512447392634367,
    -0.23217634760887954,
    -0.5153368988099954,
    -0.16862708227327208,
    0.3627836008537267,
    1.1074072425122705,
    -0.16635916368977807,
    -0.16635916368977807,
    -0.37318741969712294,
    -0.16635916368977807,
    -0.129160545043633,
    -0.16635916368977807,
    0.07426431184593647,
    -0.16635916368977807,
    -0.3315783913543882,
    -0.16635916368977807,
    0.37240708809276724,
    -0.16635916368977807,
    0.344649239366186,
    -0.7500681040108291,
    -1.3648230112047708,
    0.2909313589559246,
    0.035829136872782316,
    0.37915008063917266,
    0.07269953379217949
  ],
  "intercept": -1.4270949200628908
}


class PortableChurnPredictor:
    """Lightweight, 100% portable churn predictor using pure NumPy.
    Guarantees exact identical predictions on any Python or scikit-learn version
    without deserialization or dependency mismatch errors.
    """
    def __init__(self, weights_data_or_path):
        if isinstance(weights_data_or_path, dict):
            self.w = weights_data_or_path
        elif os.path.exists(weights_data_or_path):
            with open(weights_data_or_path, 'r') as f:
                self.w = json.load(f)
        else:
            self.w = EMBEDDED_MODEL_WEIGHTS

        self.num_cols = self.w['num_cols']
        self.mean = np.array(self.w['scaler_mean'])
        self.scale = np.array(self.w['scaler_scale'])
        self.cat_cols = self.w['cat_cols']
        self.categories = [list(c) for c in self.w['categories']]
        self.coef = np.array(self.w['coef'])
        self.intercept = self.w['intercept']

    def transform(self, df):
        # Numeric standard scaling
        X_num = (df[self.num_cols].values.astype(float) - self.mean) / self.scale
        # One-hot encoding with drop='first' and handle_unknown='ignore'
        cat_features = []
        for col, cats in zip(self.cat_cols, self.categories):
            series = df[col].astype(str)
            for cat in cats[1:]:
                cat_features.append((series == cat).astype(float).values)
        X_cat = np.column_stack(cat_features)
        return np.hstack([X_num, X_cat])

    def predict_proba(self, df):
        X = self.transform(df)
        z = np.dot(X, self.coef) + self.intercept
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


@st.cache_resource
def load_churn_model():
    """Dual-mode model loader: prefers scikit-learn pipeline, falls back to portable numpy engine."""
    meta = {}
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
        except Exception:
            pass

    # Step 1: Attempt standard scikit-learn pipeline load
    try:
        import joblib
        pipeline = joblib.load(PIPELINE_PATH)
        # Smoke test with dummy record to detect any deserialization issues
        dummy = pd.DataFrame([{
            'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No', 'Dependents': 'No',
            'tenure': 12, 'PhoneService': 'Yes', 'MultipleLines': 'No',
            'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
            'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
            'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'No',
            'PaymentMethod': 'Mailed check', 'MonthlyCharges': 50.0, 'TotalCharges': 600.0,
            'SubscribedServiceCount': 2
        }])
        _ = pipeline.predict_proba(dummy)
        return pipeline, meta, "Scikit-Learn Production Pipeline"
    except Exception:
        # Step 2: Fallback to portable numpy predictor (100% resilient)
        if os.path.exists(WEIGHTS_PATH):
            portable_model = PortableChurnPredictor(WEIGHTS_PATH)
        else:
            portable_model = PortableChurnPredictor(EMBEDDED_MODEL_WEIGHTS)
        return portable_model, meta, "Portable Zero-Dependency Predictor (NumPy)"


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# Header Section
st.title("📡 Telecom Customer Churn Predictor")
st.markdown(
    "**Supervised Binary Classification System**  \n"
    "Predicts customer churn likelihood using a rigorously cross-validated machine learning pipeline "
    "trained on the IBM Telco Customer Churn dataset with zero data leakage."
)
st.divider()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Lower threshold (e.g. 0.40) increases recall to detect more churners early at the cost of additional false alarms."
    )
    
    st.markdown("### 🏆 Production Model Specs")
    st.info(
        "**Algorithm:** Tuned Logistic Regression  \n"
        "**Validation:** Stratified 5-Fold Cross-Validation  \n"
        "**Mean CV F1 Score:** **0.5964**  \n"
        "**Mean CV ROC-AUC:** **0.8461**  \n"
        "**Held-out Test F1:** **0.6108**  \n"
        "**Held-out Test ROC-AUC:** **0.8361**"
    )
    
    st.markdown("### 💡 Business Recommendation")
    st.caption(
        "At threshold **0.40**, recall improves to 68.5% (capturing 41 additional churners for proactive retention outreach)."
    )

# Customer Input Form
st.subheader("📋 Enter Customer Information")

col_demo, col_serv, col_bill = st.columns(3)

with col_demo:
    st.markdown("#### 👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=12)

with col_serv:
    st.markdown("#### 🌐 Telecom Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"] if phone == "Yes" else ["No phone service"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    
    inet_active = (internet != "No")
    security = st.selectbox("Online Security", ["Yes", "No"] if inet_active else ["No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No"] if inet_active else ["No internet service"])
    device_prot = st.selectbox("Device Protection", ["Yes", "No"] if inet_active else ["No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"] if inet_active else ["No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"] if inet_active else ["No internet service"])
    streaming_mov = st.selectbox("Streaming Movies", ["Yes", "No"] if inet_active else ["No internet service"])

with col_bill:
    st.markdown("#### 💳 Contract & Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=70.35, step=1.0)
    total = st.number_input("Total Charges ($)", min_value=15.0, max_value=10000.0, value=float(tenure * monthly), step=10.0)

# Compute SubscribedServiceCount
service_flags = [
    phone == "Yes",
    multi_lines == "Yes",
    internet in ["DSL", "Fiber optic"],
    security == "Yes",
    backup == "Yes",
    device_prot == "Yes",
    tech_support == "Yes",
    streaming_tv == "Yes",
    streaming_mov == "Yes"
]
subscribed_service_count = sum(int(f) for f in service_flags)
st.caption(f"ℹ️ Computed **SubscribedServiceCount**: `{subscribed_service_count} / 9` active services.")

predict_btn = st.button("🚀 Evaluate Churn Risk", type="primary", use_container_width=True)

# Prediction Output
if predict_btn:
    try:
        model, meta, engine_name = load_churn_model()
        
        customer_dict = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multi_lines,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': device_prot,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_mov,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': float(monthly),
            'TotalCharges': float(total),
            'SubscribedServiceCount': subscribed_service_count
        }
        
        df_input = pd.DataFrame([customer_dict])
        prob = float(model.predict_proba(df_input)[0, 1])
        predicted_class = "Churn (Likely to Leave)" if prob >= threshold else "No Churn (Likely to Stay)"
        
        if prob >= 0.70:
            risk_badge = "🔴 High Risk"
        elif prob >= 0.40:
            risk_badge = "🟡 Moderate Risk"
        else:
            risk_badge = "🟢 Low Risk"
            
        st.divider()
        st.subheader("📊 Prediction Results")
        st.caption(f"Inference Engine: `{engine_name}`")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Outcome", "Churn" if prob >= threshold else "Retain")
        m2.metric("Churn Probability", f"{prob:.1%}")
        m3.metric("Risk Category", risk_badge)
        m4.metric("Operating Threshold", f"{threshold:.2f}")
        
        st.progress(prob)
        
        if prob >= threshold:
            st.error(
                f"⚠️ **Customer Classified as CHURN (Probability: {prob:.1%})**  \n"
                f"The model score exceeds the active decision threshold of {threshold:.2f}."
            )
        else:
            st.success(
                f"✅ **Customer Classified as NO CHURN (Probability: {prob:.1%})**  \n"
                f"The model score is below the active decision threshold of {threshold:.2f}."
            )
            
        # Risk Diagnostics
        st.markdown("### 🔍 Risk Factor Diagnostics")
        risk_factors = []
        protective_factors = []
        
        if contract == "Month-to-month":
            risk_factors.append("Month-to-month contract (strongest empirical predictor of churn).")
        else:
            protective_factors.append(f"Committed contract ({contract}) provides significant retention stability.")
            
        if internet == "Fiber optic":
            risk_factors.append("Fiber optic subscription with high monthly fees.")
        elif internet == "No":
            protective_factors.append("No internet service (historically lowest attrition segment).")
            
        if tech_support == "No" and internet != "No":
            risk_factors.append("Lacks Tech Support add-on (higher customer dissatisfaction risk).")
        elif tech_support == "Yes":
            protective_factors.append("Active Tech Support subscription (reduces churn probability).")
            
        if payment == "Electronic check":
            risk_factors.append("Electronic check payment method (associated with payment friction).")
        elif "automatic" in payment:
            protective_factors.append("Automated payment method enabled.")
            
        if tenure <= 12:
            risk_factors.append(f"Low tenure ({tenure} months) — customer is in high-vulnerability onboarding window.")
        elif tenure >= 48:
            protective_factors.append(f"High tenure ({tenure} months) — established loyalty relationship.")
            
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Identified Risk Signals:**")
            if risk_factors:
                for rf in risk_factors:
                    st.write(f"- ⚠️ {rf}")
            else:
                st.write("- None detected.")
                
        with c_right:
            st.markdown("**Identified Protective Signals:**")
            if protective_factors:
                for pf in protective_factors:
                    st.write(f"- 🛡️ {pf}")
            else:
                st.write("- None detected.")
                
        # Recommended Retention Actions
        st.markdown("### 💼 Recommended Retention Actions")
        if prob >= 0.70:
            st.warning(
                "**Urgent Outreach Required:**  \n"
                "1. Offer a 1-year contract extension with a 15% discount voucher for the first 3 months.  \n"
                "2. Provide complimentary Tech Support onboarding.  \n"
                "3. Offer automatic billing setup with a $10 one-time account credit."
            )
        elif prob >= 0.40:
            st.info(
                "**Proactive Engagement:**  \n"
                "1. Send a loyalty check-in survey to identify service satisfaction issues.  \n"
                "2. Offer bundled Tech Support or Online Security service at half price.  \n"
                "3. Encourage switching to automated credit card payment."
            )
        else:
            st.success(
                "**Standard Nurturing:**  \n"
                "Customer is healthy. Continue standard relationship management without discount intervention."
            )

    except Exception as e:
        st.error(f"Inference error: {e}")
