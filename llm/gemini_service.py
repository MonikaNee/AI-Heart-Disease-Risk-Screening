import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def generate_health_explanation(patient_data, prediction, probability):

    if prediction == 1:
        risk_level = "higher predicted risk"
    else:
        risk_level = "lower predicted risk"

    prompt = f"""
You are an AI health education assistant.

A machine learning screening model analyzed the following health information:

Age: {patient_data["age"]}
Sex: {patient_data["sex"]}
Chest pain type: {patient_data["cp"]}
Resting blood pressure: {patient_data["trestbps"]}
Cholesterol: {patient_data["chol"]}
Fasting blood sugar: {patient_data["fbs"]}
Resting ECG: {patient_data["restecg"]}
Maximum heart rate: {patient_data["thalach"]}
Exercise-induced angina: {patient_data["exang"]}
ST depression: {patient_data["oldpeak"]}
Slope: {patient_data["slope"]}
Major vessels: {patient_data["ca"]}
Thal: {patient_data["thal"]}

The machine learning model predicted:
Risk level: {risk_level}
Risk probability: {probability:.2%}

Provide:

1. A simple explanation of the model result.
2. General health guidance.
3. A reminder to consult a qualified healthcare professional when appropriate.

Important:
- Do NOT diagnose the person.
- Do NOT claim that the person has heart disease.
- Explain that this is only an ML-based screening result.
- Use simple language.
- Keep the response concise.
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    return response.text