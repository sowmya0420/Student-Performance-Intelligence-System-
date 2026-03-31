import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

def show_phase2_prediction(model, scaler, columns, background):
    st.header("Phase 2: Predictive Forecasting & Explainable AI (XAI)")
    
    st.markdown("""
    **Objective:** Transition from broad classification to precise numeric forecasting. 
    Using **Linear Regression** and **SHAP**, we quantify the exact impact of socio-academic factors on expected scores.
    """)

    # --- FLOW LOGIC: Check if data exists from Phase 1 ---
    if 'current_student_data' in st.session_state:
        st.info(f"💡 System synchronized with Phase 1 data. Status: **{st.session_state['current_classification']}**")
        saved_data = st.session_state['current_student_data']
    else:
        st.warning("⚠️ No data found from Phase 1. Using default baseline values.")
        saved_data = None

    tab1, tab2 = st.tabs(["📊 Local Score Explanation", "🌍 Global Model Behavior"])
    explainer = shap.LinearExplainer(model, background)
    
    with tab1:
        # =============================
        # Dynamic Inputs (Pre-filled from Phase 1)
        # =============================
        with st.expander("🔍 Refine Academic & Background Factors", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Academic Performance**")
                attendance = st.slider("Attendance (%)", 0, 100, int(saved_data['Attendance'].iloc[0]) if saved_data is not None else 75)
                hours_studied = st.slider("Hours Studied/Week", 0, 45, int(saved_data['Hours_Studied'].iloc[0]) if saved_data is not None else 20)
                previous_scores = st.number_input("Previous Scores", 0, 100, int(saved_data['Previous_Scores'].iloc[0]) if saved_data is not None else 60)
            
            with c2:
                st.markdown("**Environment**")
                tutoring_sessions = st.number_input("Tutoring Sessions", 0, 20, int(saved_data['Tutoring_Sessions'].iloc[0]) if saved_data is not None else 2)
                access_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)
                parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
            
            with c3:
                st.markdown("**Demographics**")
                family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
                motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)

        if st.button("Calculate Forecasted Score", type="primary"):
            # Construct Input
            input_data = {
                "Attendance": attendance, "Hours_Studied": hours_studied,
                "Previous_Scores": previous_scores, "Tutoring_Sessions": tutoring_sessions,
                "Access_to_Resources": access_resources, "Parental_Involvement": parental_involvement,
                "Family_Income": family_income, "Motivation_Level": motivation_level
            }
            
            input_df = pd.get_dummies(pd.DataFrame([input_data])).reindex(columns=columns, fill_value=0)
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            # Display Prediction
            st.metric("Forecasted Exam Score", f"{round(prediction, 2)} Points")

            # SHAP Visualization Logic (Grouped for readability)
            # SHAP Visualization Logic (Grouped for readability)
            shap_explanation = explainer(input_scaled)
            shap_values = shap_explanation.values[0]

            # Define categorical base features
            categorical_features = [
                "Access_to_Resources",
                "Parental_Involvement",
                "Family_Income",
                "Motivation_Level"
            ]

            # Group SHAP values
            grouped_shap = {}
            grouped_data = {}

            for feature, shap_val, data_val in zip(columns, shap_values, input_scaled[0]):

                grouped = False

                for cat in categorical_features:
                    if feature.startswith(cat + "_"):
                        if cat not in grouped_shap:
                            grouped_shap[cat] = 0
                            grouped_data[cat] = 1  # placeholder
                        grouped_shap[cat] += shap_val
                        grouped = True
                        break

                if not grouped:
                    grouped_shap[feature] = shap_val
                    grouped_data[feature] = data_val

            # Convert grouped results
            final_features = list(grouped_shap.keys())
            final_shap_values = np.array(list(grouped_shap.values()))
            final_data = np.array(list(grouped_data.values()))

            # Waterfall Plot
            st.subheader("Waterfall Analysis: Contribution to Score")
            fig = plt.figure()

            shap.plots.waterfall(
                shap.Explanation(
                    values=final_shap_values,
                    base_values=explainer.expected_value,
                    data=final_data,
                    feature_names=final_features
                ),
                show=False
            )

            st.pyplot(fig)
                        
            # Update Session State for Phase 3 (Counterfactuals)
            st.session_state['forecasted_score'] = prediction
            st.session_state['regression_input'] = input_df
            
            st.success("✅ Forecast complete. Proceed to **Phase 3** for Risk Mitigation & Guidance.")

    with tab2:
        st.write("This section visualizes how the model weights these factors across the entire student population.")
        st.subheader("Global Feature-Level Analysis")

        # Convert background to DataFrame
        if isinstance(background, np.ndarray):
            X_background = pd.DataFrame(background, columns=columns)
        else:
            X_background = background.copy()

        # Use LinearExplainer (since model is linear)
        explainer_global = shap.LinearExplainer(model, X_background)
        shap_values_global = explainer_global.shap_values(X_background)

        # =================================================
        # Global Feature Importance
        # =================================================
        st.subheader("Global Feature Importance")

        mean_shap = np.abs(shap_values_global).mean(axis=0)

        importance_df = pd.DataFrame({
            "Feature": X_background.columns,
            "Mean SHAP Value": mean_shap
        }).sort_values(by="Mean SHAP Value", ascending=False)

        # Show only top 8 to avoid clutter
        top_features = importance_df.head(8).reset_index(drop=True)
        top_features.index += 1
        top_features.index.name = "Rank"

        # Two-column layout (table + bar chart)
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("### Top Influential Features")
            st.table(top_features)

        with col2:
            fig1, ax1 = plt.subplots(figsize=(5.5, 3.8))
            ax1.barh(top_features["Feature"][::-1], 
                    top_features["Mean SHAP Value"][::-1])
            ax1.set_xlabel("Mean SHAP Value")
            st.pyplot(fig1)
            plt.close(fig1)

        # Interpretation
        top_feature = top_features.iloc[0]["Feature"]
        top_value = top_features.iloc[0]["Mean SHAP Value"]

        st.markdown(f"""
        ### Interpretation:
        - The most influential feature globally is **{top_feature}**.
        - It has an average SHAP magnitude of **{round(top_value, 3)}**.
        - This indicates the model relies heavily on this factor across the dataset.
        """)

        st.markdown("---")

        # =================================================
        # SHAP Summary Plot (Centered)
        # =================================================
        st.subheader("SHAP Summary Plot")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5.8, 3.8))
            shap.summary_plot(
                shap_values_global,
                X_background,
                show=False
            )
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("""
        ### Insights:
        - Each dot represents a student.
        - Red indicates high feature value.
        - Blue indicates low feature value.
        - Right side increases predicted score.
        - Left side decreases predicted score.
        - Since the model is linear, relationships appear proportional and monotonic.
        """)

        st.markdown("---")

        # =================================================
        # Directional Influence
        # =================================================
        shap_df = pd.DataFrame(shap_values_global, columns=X_background.columns)

        positive_impact = shap_df.mean().sort_values(ascending=False)
        negative_impact = shap_df.mean().sort_values()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Strongest Positive Drivers")
            st.table(positive_impact.head(5))

        with col2:
            st.subheader("Strongest Negative Drivers")
            st.table(negative_impact.head(5))

        top_positive = positive_impact.index[0]
        top_negative = negative_impact.index[0]

        st.markdown(f"""
        ### Directional Interpretation:
        - **{top_positive}** contributes the strongest positive impact on exam score predictions.
        - **{top_negative}** contributes the strongest negative impact on exam score predictions.
        - These insights help identify which academic factors boost performance and which may hinder it.
        """)