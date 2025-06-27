import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Logistic Regression Predictor")

st.write("Enter the values for prediction:")

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])

sex = st.selectbox("Sex", ["male", "female"])
sex_encoded = 1 if sex == "male" else 0

age = st.slider("Age", 0, 100, 25)



# Combine into model input format
input_data = np.array([[pclass, sex_encoded, age]])

# Predict on button click
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    output = "Survived" if prediction[0] == 1 else "Did not survive"
    st.success(f"Prediction: {output}")
