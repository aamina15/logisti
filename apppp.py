import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Insurance Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Insurance Prediction")
st.write("Predict whether a person will buy insurance or not using Logistic Regression.")

# -----------------------------------
# Load Dataset
# -----------------------------------
df = pd.read_csv("insurance_data.csv")

st.subheader("Dataset")
st.dataframe(df)

# -----------------------------------
# Train Model
# -----------------------------------
X = df[["age"]]
y = df["bought_insurance"]

model = LogisticRegression()
model.fit(X, y)

# -----------------------------------
# User Input
# -----------------------------------
st.subheader("Enter Person's Age")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict"):

    prediction = model.predict([[age]])[0]
    probability = model.predict_proba([[age]])[0][1]

    if prediction == 1:
        st.success("✅ This person is likely to BUY insurance.")
    else:
        st.error("❌ This person is NOT likely to buy insurance.")

    st.write(f"Probability of buying insurance: **{probability:.2%}**")

# -----------------------------------
# Model Information
# -----------------------------------
st.subheader("Model Details")

st.write("Coefficient:", model.coef_[0][0])
st.write("Intercept:", model.intercept_[0])
