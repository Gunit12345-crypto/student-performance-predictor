import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓"
)

# Title
st.title("🎓 Student Performance Predictor")

# Instructions
st.info("""
📌 PASS Criteria:

✅ Overall Percentage must be 40% or more

AND

✅ Student must score at least 30 marks in each subject
""")

st.write("""
Enter student marks to predict whether
the student will PASS or FAIL.
""")

st.divider()

# Score inputs
math_score = st.slider(
    "📘 Math Score",
    0,
    100,
    50
)

reading_score = st.slider(
    "📖 Reading Score",
    0,
    100,
    50
)

writing_score = st.slider(
    "✍️ Writing Score",
    0,
    100,
    50
)

# Average score
average_score = (
    math_score +
    reading_score +
    writing_score
) / 3

st.success(f"📊 Average Percentage: {average_score:.2f}%")

# Create dataframe
input_data = pd.DataFrame({
    'math score': [math_score],
    'reading score': [reading_score],
    'writing score': [writing_score]
})

# Scale input
input_scaled = scaler.transform(input_data)

st.divider()

# Predict button
if st.button("🔍 Predict Result"):

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:

        st.success("✅ Student is likely to PASS")

    else:

        st.error("❌ Student is likely to FAIL")

# Footer
st.divider()

st.caption("Machine Learning Project using Logistic Regression and KNN")