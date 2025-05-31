import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load("models/final_rf_model-copy1.pkl")
scaler = joblib.load("models/final_scaler.pkl")
encoders = joblib.load("models/final_encoders.pkl")  # Dictionary of LabelEncoders
label_encoder = joblib.load("models/final_label_encoder.pkl")  # For decoding labels if needed

st.title("🚨 Intrusion Detection System")
st.markdown("Upload a CSV file with network traffic data to detect intrusions.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV File", type="csv")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.write("📊 Uploaded Data")
    st.dataframe(input_df)

    # Handle encoding for categorical columns
    for col in ['protocol_type', 'service', 'flag']:
        known_labels = encoders[col].classes_

        # Replace unseen labels with '__unknown__'
        input_df[col] = input_df[col].apply(lambda x: x if x in known_labels else '__unknown__')

        # Add '__unknown__' to encoder classes if missing
        if '__unknown__' not in encoders[col].classes_:
            new_classes = np.append(encoders[col].classes_, '__unknown__')
            encoders[col].classes_ = new_classes

        # Encode
        input_df[col] = encoders[col].transform(input_df[col])

    # Normalize numeric features
    input_scaled = scaler.transform(input_df)

    # Predict
    predict_btn = st.button("🚀 Predict Intrusion")
    if predict_btn:
        predictions = model.predict(input_scaled)
        results = ["Attack" if p == 0 else "Normal" for p in predictions]
        input_df["Prediction"] = results

        st.success("✅ Prediction Complete!")
        st.dataframe(input_df)
