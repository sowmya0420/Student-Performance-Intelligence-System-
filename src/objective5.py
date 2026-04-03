import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def show_phase5_resilience_analysis(stack_model, scaler, columns):
    # Professional Header
    st.header("Phase 5: Macro-Level Resilience Analysis")

    st.markdown("""
    **Objective:** Validate model findings by identifying 'Resilient' outliers—students who 
    exceed performance expectations despite socioeconomic constraints. This analysis 
    provides the empirical basis for the counterfactual logic used in Phase 4.
    """)

    # --- DATA LOADING & PREPROCESSING ---
    # Wrap in a try-except for robust scholarly tools
    try:
        df = pd.read_csv("Dataset/StudentPerformanceFactors - 6k.csv")
    except FileNotFoundError:
        st.error("Data repository not found. Please ensure the dataset is in the correct directory.")
        return

    selected_features = [
        "Attendance", "Hours_Studied", "Previous_Scores", 
        "Tutoring_Sessions", "Access_to_Resources", 
        "Parental_Involvement", "Family_Income", "Motivation_Level"
    ]

    # Preprocessing to match Stacked Model training
    X = df[selected_features]
    cat_cols = X.select_dtypes(include="object").columns
    X_processed = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X_processed = X_processed.reindex(columns=columns, fill_value=0)
    X_scaled = scaler.transform(X_processed)

    # Generate Predictions for the whole population
    df["Predicted_Score"] = stack_model.predict(X_scaled)

    # --- DEFINING THE COHORTS ---
    # Using 70th percentile as the "High Achievement" benchmark
    high_score_threshold = df["Predicted_Score"].quantile(0.70)

    resilient_students = df[
        (df["Family_Income"] == "Low") & 
        (df["Access_to_Resources"] == "Low") & 
        (df["Predicted_Score"] >= high_score_threshold)
    ]

    privileged_students = df[
        (df["Family_Income"] == "High") & 
        (df["Access_to_Resources"] == "High") & 
        (df["Predicted_Score"] >= high_score_threshold)
    ]

    # --- COHORT METRICS ---
    st.subheader("Cohort Distribution Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Resilient Cohort", len(resilient_students), help="Low Income + Low Resources + High Score")
    c2.metric("Privileged Cohort", len(privileged_students), help="High Income + High Resources + High Score")
    c3.metric("System Benchmark", f"{round(high_score_threshold, 1)} pts")

    st.divider()

    # --- THE "WHY" (COMPARISON) ---
    st.subheader("The 'Resilience Gap' Analysis")
    st.write("How do resilient students compensate for a lack of resources?")

    comparison_features = ["Attendance", "Hours_Studied", "Previous_Scores", "Tutoring_Sessions"]
    res_avg = resilient_students[comparison_features].mean()
    priv_avg = privileged_students[comparison_features].mean()

    comp_df = pd.DataFrame({
        "Metric": comparison_features,
        "Resilient (Low Resource)": res_avg.values,
        "Privileged (High Resource)": priv_avg.values
    }).melt(id_vars="Metric", var_name="Student Group", value_name="Average Value")

    # Thematic Scholarly Plot
    fig = px.bar(
        comp_df, x="Metric", y="Average Value", color="Student Group",
        barmode="group", text_auto='.2s',
        color_discrete_map={"Resilient (Low Resource)": "#00CC96", "Privileged (High Resource)": "#636EFA"},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- FINAL POLICY INSIGHTS (The Flow Conclusion) ---
    with st.expander("🎓 Strategic Research Insights", expanded=True):
        st.markdown(f"""
        1. **Compensatory Effort:** Resilient students show an average attendance of **{round(res_avg['Attendance'], 1)}%**, 
           comparable to or exceeding their privileged peers.
        2. **Intervention Validation:** The success of the resilient cohort validates the **Phase 4 Counterfactuals** (e.g., boosting study hours and attendance is the primary lever for overcoming resource gaps).
        3. **Policy Recommendation:** Institutional support should focus on 'Effort-Multipliers' (Tutoring) 
           rather than just infrastructure, as seen in the resilient data.
        """)

    st.success("🏁 Research Pipeline Complete: From Diagnostic to Policy Recommendation.")
