# Import libraries
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the fraud detection model
try:
    with open('fraud_model.sav', 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Ensure 'fraud_model.pkl' is in the working directory.")
    st.stop()

# Prediction function with probabilities
def predict_fraud(input_data):
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    prediction_proba = loaded_model.predict_proba(input_array)
    return prediction[0], prediction_proba[0]

# Main app
def main():
    st.title("💳 Credit Card Fraud Detection App")
    st.markdown("Enter transaction details to determine if it is **fraudulent** or **legitimate**.")

    features = [
        'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
        'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
        'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
    ]

    # Create input form
    user_input = []
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            val = st.number_input(f"{feature}", value=0.0, format="%.4f", help=f"Enter value for {feature}")
            user_input.append(val)

    # Validate input before prediction
    if any(val is None for val in user_input):
        st.warning("⚠️ Please fill in all input fields.")
        return

    if st.button("Detect Fraud"):
        prediction, prediction_proba = predict_fraud(user_input)

        # Display result
        if prediction == 1:
            st.error("⚠️ This transaction is likely **fraudulent**.")
        else:
            st.success("✅ This transaction appears to be **legitimate**.")

        # Show prediction probability
        st.markdown("### 🔎 Prediction Confidence")
        fraud_prob = prediction_proba[1] * 100
        legit_prob = prediction_proba[0] * 100
        st.info(f"Legitimate: **{legit_prob:.2f}%**, Fraudulent: **{fraud_prob:.2f}%**")

        # Plot probabilities
        st.markdown("### 📊 Probability Distribution")
        st.bar_chart({
            "Prediction Probability": {
                "Legitimate": legit_prob,
                "Fraudulent": fraud_prob
            }
        })

# Run the app
if __name__ == "__main__":
    main()
