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

sibsp = st.number_input("Number of Siblings/Spouse Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)

fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=32.0)

embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]

# Combine into model input format
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Predict on button click
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    output = "Survived" if prediction[0] == 1 else "Did not survive"
    st.success(f"Prediction: {output}")
