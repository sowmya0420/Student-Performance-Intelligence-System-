import streamlit as st
import pandas as pd
import numpy as np

def show_phase1_stratification(model, scaler, features):
    # Scholarly Heading to match the new flow
    st.header("Phase 1: Performance Stratification & Diagnostic Profiling")
    
    st.markdown("""
    **Objective:** Establish a baseline classification of the student cohort. 
    This diagnostic phase categorizes students into performance tiers to prioritize intervention.
    """)

    # ==========================
    # Student Inputs (Organized in Columns for a "Scholarly" look)
    # ==========================
    with st.expander("📝 Input Student Metrics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            attendance = st.slider("Attendance (%)", 0, 100, 75, help="Historical attendance record.")
            hours_studied = st.slider("Weekly Study Hours", 0, 45, 20)
            sleep_hours = st.slider("Daily Sleep Hours", 0, 12, 7)
        with col2:
            previous_scores = st.number_input("Previous Academic Scores", 0, 100, 60)
            tutoring_sessions = st.number_input("Monthly Tutoring Sessions", 0, 20, 2)
            # Placeholder for potential categorical data if your model uses it
            participation = st.selectbox("Class Participation", ["High", "Medium", "Low"])

    # ==========================
    # Prediction Logic
    # ==========================
    if st.button("Generate Diagnostic Profile", type="primary"):
        input_data = {
            "Attendance": attendance,
            "Hours_Studied": hours_studied,
            "Sleep_Hours": sleep_hours,
            "Previous_Scores": previous_scores,
            "Tutoring_Sessions": tutoring_sessions
        }

        input_df = pd.DataFrame([input_data])
        
        # Ensure column order matches training features
        input_df = input_df[features]

        # Scaling and Prediction
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        st.divider()
        
        # Displaying Results with flow-oriented language
        st.subheader("Diagnostic Results")
        
        if prediction == 1:
            st.error("⚠️ The student is likely to perform **Poorly (At Risk)**.")
            st.error("### Status: **At-Risk Tier**")
            st.warning("Immediate intervention recommended. Proceed to **Phase 2** for score forecasting.")
        else:
            st.success("✅ The student is likely to perform **Well**.")
            st.success("### Status: **Optimal Performance Tier**")
            st.info("Student is meeting baseline expectations.")

        # Metric visualization for scholarly impact
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Classification Confidence", f"{round(np.max(prob)*100, 2)}%")
        m_col2.metric("Risk Probability", f"{round(prob[1]*100, 2)}%")

        # CRITICAL: Saving to Session State to create "Flow"
        # This allows Objective 2 (Regression) to "know" what happened here.
        st.session_state['current_student_data'] = input_df
        st.session_state['current_classification'] = "At-Risk" if prediction == 1 else "Optimal"
        
        st.info("Data cached. You can now navigate to **Phase 2: Predictive Forecasting**.")