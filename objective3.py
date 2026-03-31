import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_phase3_risk_detection(model, scaler, features):
    st.header("Phase 3: Diagnostic Risk Stratification")
    st.markdown("""
    **Objective:** Identify students in the 'At-Risk' zone using ensemble learning (Random Forest).
    This phase acts as an automated 'early warning system' for educators.
    """)

    # --- FLOW: Inherit from previous Phase 2 if exists ---
    init_vals = [75, 4, 7, 60, 2] # Defaults
    if 'regression_input' in st.session_state:
        df = st.session_state['regression_input']
        # Mapping common features from the Regression module
        init_vals = [
            int(df['Attendance'].iloc[0]), 
            int(df['Hours_Studied'].iloc[0]), 
            7, # Default Sleep
            int(df['Previous_Scores'].iloc[0]), 
            int(df['Tutoring_Sessions'].iloc[0])
        ]

    # =========================
    # Diagnostic Input
    # =========================
    with st.expander("📝 Audit Student Metrics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            attendance = st.slider("Attendance (%)", 0, 100, init_vals[0])
            hours_studied = st.slider("Study Hours/Day", 0, 12, init_vals[1])
            sleep_hours = st.slider("Sleep Hours/Day", 0, 12, init_vals[2])
        with col2:
            previous_scores = st.slider("Previous Exam Score", 0, 100, init_vals[3])
            tutoring_sessions = st.slider("Tutoring Sessions/Month", 0, 20, init_vals[4])

    if st.button("Run Risk Assessment", type="primary"):
        input_data = pd.DataFrame([[
            attendance, hours_studied, sleep_hours, 
            previous_scores, tutoring_sessions
        ]], columns=features)

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        # SAVE TO SESSION FOR PHASE 4
        st.session_state['at_risk_status'] = prediction
        st.session_state['risk_prob'] = prob
        st.session_state['risk_input_data'] = {
            "Attendance": attendance, "Hours_Studied": hours_studied,
            "Sleep_Hours": sleep_hours, "Previous_Scores": previous_scores,
            "Tutoring_Sessions": tutoring_sessions
        }

        st.divider()
        if prediction == 1:
            st.error(f"### ⚠️ Status: HIGH RISK ({round(prob*100, 2)}%)")
            st.info("The student profile matches historical failure patterns. **Proceed to Phase 4 for intervention.**")
        else:
            st.success(f"### ✅ Status: LOW RISK ({round(prob*100, 2)}%)")

        # Global Feature Importance (Scholarly Context)
        st.subheader("Global Model Logic (Ensemble Insights)")
        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(importance_df["Feature"], importance_df["Importance"], color='#1f77b4')
        st.pyplot(fig)