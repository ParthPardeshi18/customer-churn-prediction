import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

@st.cache_resource
def load_artifacts():
    with open(os.path.join(OUTPUT_DIR, "churn_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR, "feature_list.pkl"), "rb") as f:
        feature_list = pickle.load(f)
    return model, scaler, feature_list

model, scaler, feature_list = load_artifacts()

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Predictor")
st.markdown("Predict whether a telecom customer is likely to churn based on their account details.")
st.divider()

with st.sidebar:
    st.header("Customer Details")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 200.0, 50.0, step=5.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    st.divider()
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

if predict_btn:
    input_data = pd.DataFrame(np.zeros((1, len(feature_list))), columns=feature_list)

    total_charges = monthly_charges * tenure if tenure > 0 else monthly_charges
    charges_per_tenure = monthly_charges / (tenure + 1)

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges
    input_data["charges_per_tenure"] = charges_per_tenure

    continuous_cols = ["tenure", "MonthlyCharges", "TotalCharges", "charges_per_tenure"]
    input_data[continuous_cols] = scaler.transform(input_data[continuous_cols])

    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    if internet_service == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif internet_service == "No":
        input_data["InternetService_No"] = 1

    if tech_support == "Yes":
        input_data["TechSupport_Yes"] = 1
    elif tech_support == "No internet service":
        input_data["TechSupport_No internet service"] = 1

    if online_security == "Yes":
        input_data["OnlineSecurity_Yes"] = 1
    elif online_security == "No internet service":
        input_data["OnlineSecurity_No internet service"] = 1

    if tenure <= 12:
        pass
    elif tenure <= 24:
        input_data["tenure_group_13-24"] = 1
    elif tenure <= 48:
        input_data["tenure_group_25-48"] = 1
    else:
        input_data["tenure_group_49-72"] = 1

    prob = model.predict_proba(input_data)[0][1]
    prob_pct = prob * 100

    if prob_pct > 70:
        risk = "High"
        color = "#e74c3c"
        recommendation = "⚠️ **Immediate action needed.** Consider offering a discount, contract upgrade, or dedicated support to retain this customer."
    elif prob_pct > 40:
        risk = "Medium"
        color = "#f39c12"
        recommendation = "🔔 **Monitor closely.** Proactive outreach with loyalty offers or service upgrades could reduce churn risk."
    else:
        risk = "Low"
        color = "#27ae60"
        recommendation = "✅ **Customer appears satisfied.** Continue delivering quality service and periodic check-ins."

    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{prob_pct:.1f}%")
    with col2:
        st.markdown(f"### Risk: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)

    st.progress(float(min(prob, 1.0)))
    st.divider()
    st.markdown("### Recommendation")
    st.markdown(recommendation)
else:
    st.info("👈 Set customer details in the sidebar and click **Predict Churn** to see results.")
