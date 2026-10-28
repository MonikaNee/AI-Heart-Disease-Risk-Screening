import streamlit as st
import pandas as pd
import joblib

from llm.gemini_service import generate_health_explanation

# Load trained model
model = joblib.load("model/heart_disease_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Screening",
    page_icon="❤️",
    layout="wide"
)

# Title
st.title("❤️ AI-Based Heart Disease Risk Screening")

st.write(
    "Enter the health information below to estimate the predicted risk "
    "using a machine learning model and receive an AI-generated explanation."
)

st.warning(
    "This application is for screening and educational purposes only. "
    "It is not a medical diagnosis."
)

# -----------------------------
# INPUT SECTION
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Personal Information")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=55
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.number_input(
        "Chest Pain Type (0-3)",
        min_value=0,
        max_value=3,
        value=1
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50.0,
        max_value=250.0,
        value=140.0
    )

    chol = st.number_input(
        "Cholesterol Level",
        min_value=50.0,
        max_value=600.0,
        value=250.0
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    restecg = st.number_input(
        "Resting ECG Result (0-2)",
        min_value=0,
        max_value=2,
        value=1
    )


with col2:

    st.subheader("Test Information")

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50.0,
        max_value=250.0,
        value=150.0
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

    slope = st.number_input(
        "Slope (0-2)",
        min_value=0,
        max_value=2,
        value=1
    )

    ca = st.number_input(
        "Number of Major Vessels (0-3)",
        min_value=0,
        max_value=3,
        value=0
    )

    thal = st.number_input(
        "Thal Value (0-3)",
        min_value=0,
        max_value=3,
        value=2
    )


st.divider()


# -----------------------------
# PREDICTION
# -----------------------------

if st.button("🔍 Check Risk", use_container_width=True):

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # ML prediction
    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    # -----------------------------
    # RISK CATEGORY
    # -----------------------------

    if probability < 0.30:

        risk_category = "Lower Predicted Risk"

        st.success(
            "🟢 Lower Predicted Risk"
        )

    elif probability < 0.70:

        risk_category = "Moderate Predicted Risk"

        st.warning(
            "🟡 Moderate Predicted Risk"
        )

    else:

        risk_category = "Higher Predicted Risk"

        st.error(
            "🔴 Higher Predicted Risk"
        )

    # -----------------------------
    # RESULT
    # -----------------------------

    st.subheader("🩺 Screening Result")

    st.write(f"**Risk Category:** {risk_category}")

    st.metric(
        "Model-Estimated Probability",
        f"{probability:.2%}"
    )

    st.caption(
        "This percentage is the machine learning model's predicted "
        "probability for the positive-risk class. It is not a clinically "
        "validated medical probability."
    )

    # -----------------------------
    # GEMINI AI EXPLANATION
    # -----------------------------

    st.subheader("🤖 AI Health Explanation")

    patient_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    try:

        with st.spinner("Generating AI explanation..."):

            explanation = generate_health_explanation(
                patient_data,
                prediction,
                probability
            )

        st.markdown(explanation)

    except Exception:

        st.warning(
            "🤖 AI explanation is temporarily unavailable. "
            "The machine learning screening result above is still available. "
            "Please try again later."
        )

    # -----------------------------
    # MEDICAL DISCLAIMER
    # -----------------------------

    st.info(
        "⚠️ This result is generated by a machine learning model "
        "for screening and educational purposes. It is not a medical "
        "diagnosis. Consult a qualified healthcare professional "
        "for medical advice."
    )