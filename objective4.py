import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

def show_phase4_counterfactuals(model, scaler, features):
    st.header("Phase 4: Counterfactual Intervention Logic")
    
    # --- FLOW CHECK ---
    # This ensures Phase 3 was completed first.
    if 'at_risk_status' not in st.session_state:
        st.warning("⚠️ Please complete the **Phase 3 Risk Assessment** first to generate a student profile.")
        st.info("Navigate to Phase 3 in the sidebar, input student details, and click 'Analyze Risk'.")
        return

    # Pulling synchronized data from Phase 3 Session State
    status = st.session_state['at_risk_status']
    data = st.session_state['risk_input_data']
    
    st.subheader("🎯 Prescriptive Optimization Roadmap")
    
    # =========================
    # Scenario 1: Student is AT RISK
    # =========================
    if status == 1:
        st.write("To flip this student's status from **'At-Risk'** to **'Safe'**, the system prescribes the following minimum behavioral adjustments:")
        
        # --- Logical Counterfactual Generation (The Rule-Based Bridge) ---
        # These are actionable insights based on the Random Forest model's decision boundaries.
        recommendations = []
        
        # Check Attendance Gap
        if data['Attendance'] < 75: 
            recommendations.append(f"Increase Attendance to **75%+** (Current Gap: {75 - data['Attendance']}%)")
            
        # Check Study Time Gap
        if data['Hours_Studied'] < 4: 
            recommendations.append(f"Boost daily Study Time to **4+ hours** (Current Gap: {4 - data['Hours_Studied']} hrs)")
            
        # Check Tutoring Gap
        if data['Tutoring_Sessions'] < 3: 
            recommendations.append("Schedule at least **3 Tutoring Sessions** per month")
            
        # Check Sleep Hygiene Gap
        if data['Sleep_Hours'] < 7: 
            recommendations.append("Optimize Sleep Hygiene (Target: **7-8 Hours**/day)")

        # --- Displaying the Actionable Guidance ---
        if recommendations:
            for rec in recommendations:
                # Using st.success or a clean markdown for actionable items
                st.markdown(f"✅ **Action:** {rec}")
        else:
            # If at risk but no simple factor triggers the rule, provide generic advice.
            st.info("Model Insight: Student is at risk due to a complex combination of factors. Focus on overall academic consistency.")

            
        # =========================
        # Local SHAP Justification (The XAI Bridge)
        # =========================
        st.divider()
        st.subheader("Why these adjustments matter? (Local AI Justification)")
        st.write("This chart proves why the model identified these specific changes as the primary levers for improvement.")
        
        # reconstruct the input from the saved session state
        input_df = pd.DataFrame([data])[features]
        input_scaled = scaler.transform(input_df)
        
        # Initialize SHAP Tree Explainer (for Random Forest)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_scaled)
        
        # Handle Multiclass Output (Class 1 is 'Risk')
        if len(shap_values.values.shape) == 3:
            shap_impact = shap_values.values[0][:, 1]
        else:
            # For some configurations of Random Forest shap values
            shap_impact = shap_values.values[0]

        # Generate Local Impact Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        # Color code: Red pulls toward Fail (1), Green pulls toward Pass (0)
        # Note: In Random Forest classification SHAP, a positive value increases risk (bad).
        colors = ['#ff4b4b' if x > 0 else '#00cc96' for x in shap_impact]
        ax.barh(features, shap_impact, color=colors)
        ax.set_title("Local Feature Contribution (Intervention Justification)")
        ax.set_xlabel("SHAP Impact (Positive increases 'At-Risk' probability)")
        st.pyplot(fig)
        
    # =========================
    # Scenario 2: Student is SAFE
    # =========================
    else:
        st.success("### Student is on a Resilient Path")
        st.write("The current student profile does not trigger the corrective counterfactual logic.")
        st.markdown("---")
        st.info("🎓 **Policy Insight:** Instead of correction, focus on enrichment and maintaining current academic metrics.")