
import streamlit as st

import premium_theme
premium_theme.apply()

st.set_page_config(page_title="DR + Metabolic Risk", layout="wide")
st.title("🩺 Diabetic Retinopathy + Metabolic Risk Assessment")
st.caption("Research prototype — not a medical device. Not for clinical use.")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Retina Image Analysis")
    uploaded_file = st.file_uploader("Upload fundus image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        st.success("DR Grade: 2 (Moderate) — [placeholder]")
        st.write("Confidence: Grade 0: 10% | Grade 1: 15% | Grade 2: 50% | Grade 3: 20% | Grade 4: 5%")

with col2:
    st.header("2. Metabolic Risk Factors")
    hba1c = st.slider("HbA1c (%)", 4.0, 15.0, 7.0)
    bp = st.slider("Blood Pressure (systolic)", 90, 200, 120)
    duration = st.slider("Diabetes Duration (years)", 0, 30, 5)

    if st.button("Calculate Metabolic Risk"):
        risk_score = 0
        if hba1c > 8: risk_score += 2
        elif hba1c > 6.5: risk_score += 1
        if bp > 140: risk_score += 2
        elif bp > 120: risk_score += 1
        if duration > 10: risk_score += 2
        elif duration > 5: risk_score += 1

        risk_level = "Low" if risk_score <= 2 else "Medium" if risk_score <= 4 else "High"
        st.info(f"Metabolic Risk: {risk_level}")

st.header("3. Combined Risk Report")
if st.button("Generate Combined Report"):
    st.write("**Final Recommendation:** Refer to ophthalmologist within 3 months")
    st.warning("Note: Image and metabolic analysis are separate models combined at decision level — proof-of-concept.")
