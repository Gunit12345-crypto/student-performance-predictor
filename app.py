
import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main {
    background: linear-gradient(to bottom, #0E1117, #111827);
}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #4ADE80;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #D1D5DB;
    margin-bottom: 30px;
}

.card {
    background-color: #1F2937;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #374151;
    margin-bottom: 25px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.25);
}

.card-title {
    font-size: 24px;
    font-weight: bold;
    color: white;
    margin-bottom: 15px;
}

.criteria {
    font-size: 18px;
    color: #E5E7EB;
    line-height: 1.8;
}

.average-box {
    background-color: #111827;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-top: 10px;
    border: 1px solid #374151;
}

.average-text {
    font-size: 30px;
    font-weight: bold;
    color: #60A5FA;
}

.pass-box {
    background-color: #052e16;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    border: 2px solid #22c55e;
    margin-top: 20px;
}

.fail-box {
    background-color: #3b0a0a;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    border: 2px solid #ef4444;
    margin-top: 20px;
}

.result-text {
    font-size: 34px;
    font-weight: bold;
    color: white;
}

.small-text {
    font-size: 17px;
    color: #D1D5DB;
    margin-top: 10px;
}

.stButton > button {
    background: linear-gradient(to right, #22c55e, #16a34a);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    height: 60px;
    width: 100%;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #16a34a, #15803d);
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Machine Learning Project using Logistic Regression & KNN</div>',
    unsafe_allow_html=True
)

# PASS Criteria Card
st.markdown("""
<div class="card">
<div class="card-title">📌 PASS Criteria</div>
<div class="criteria">
✅ Overall Percentage must be <b>40% or more</b><br><br>
✅ Student must score at least <b>30 marks in each subject</b>
</div>
</div>
""", unsafe_allow_html=True)

# Input Card
st.markdown('<div class="card-title">📊 Enter Student Marks</div>', unsafe_allow_html=True)

math_score = st.slider("📘 Math Score", 0, 100, 50)
reading_score = st.slider("📖 Reading Score", 0, 100, 50)
writing_score = st.slider("✍️ Writing Score", 0, 100, 50)

# Average Calculation
average_score = (
    math_score +
    reading_score +
    writing_score
) / 3

# Show Average
st.markdown(f"""
<div class="average-box">
<div style="font-size:18px; color:#D1D5DB;">Average Percentage</div>
<div class="average-text">{average_score:.2f}%</div>
</div>
""", unsafe_allow_html=True)

# Create dataframe
input_data = pd.DataFrame({
    'math score': [math_score],
    'reading score': [reading_score],
    'writing score': [writing_score]
})

# Scale input
input_scaled = scaler.transform(input_data)

st.write("")

# Predict button
if st.button("🔍 Predict Student Result"):

    prediction = model.predict(input_scaled)[0]

    # PASS
    if prediction == 1:

        st.balloons()

        st.markdown("""
        <div class="pass-box">
        <div class="result-text">✅ PASS</div>
        <div class="small-text">
        Student has successfully met the passing criteria.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # FAIL
    else:

        st.markdown("""
        <div class="fail-box">
        <div class="result-text">❌ FAIL</div>
        <div class="small-text">
        Student did not meet the required passing criteria.
        </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.write("")
st.divider()

