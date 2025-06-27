import streamlit as st
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Logistic Regression Predictor")

st.write("Enter the values for prediction:")

# Example input fields (replace with your real features)
feature1 = st.number_input("Survived", step=0.1)
feature2 = st.number_input("Sex", step=0.1)

input_data = np.array([[feature1, feature2]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Class: {prediction[0]}")