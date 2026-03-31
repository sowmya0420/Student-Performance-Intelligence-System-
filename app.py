import streamlit as st
import joblib
import pandas as pd

# Import your refactored module functions
# Ensure these filenames (.py) and function names match your saved files
from objective1 import show_phase1_stratification
from objective2 import show_phase2_prediction
from objective3 import show_phase3_risk_detection
from objective4 import show_phase4_counterfactuals
from objective5 import show_phase5_resilience_analysis

# -----------------------------
# 1. PAGE CONFIG & STYLING
# -----------------------------
st.set_page_config(
    page_title="Student Intelligence System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Scholarly UI Styling
st.markdown("""
<style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1 { color: #1e3c72; font-family: 'Serif'; }
    .sidebar-text { font-size: 14px; color: #555; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. LOAD MODELS (Cached for Performance)
# -----------------------------
@st.cache_resource
def load_all_assets():
    # Regression Assets (Phase 2)
    reg_model = joblib.load("reg_model.pkl")
    reg_scaler = joblib.load("reg_scaler.pkl")
    reg_columns = joblib.load("reg_columns.pkl")
    reg_background = joblib.load("reg_background.pkl")
    
    # Classification/Risk Assets (Phase 3 & 4)
    clf_model = joblib.load("clf_model.pkl")
    clf_scaler = joblib.load("clf_scaler.pkl")
    clf_features = joblib.load("clf_features.pkl")
    
    # Logistic/Stratification Assets (Phase 1)
    logistic_model = joblib.load("logistic_model.pkl")
    logistic_scaler = joblib.load("logistic_scaler.pkl")
    logistic_features = joblib.load("logistic_features.pkl")
    
    # Stacking Model (Phase 5)
    stack_model = joblib.load("stack_model.pkl")
    
    return (reg_model, reg_scaler, reg_columns, reg_background,
            clf_model, clf_scaler, clf_features,
            logistic_model, logistic_scaler, logistic_features,
            stack_model)

assets = load_all_assets()
(reg_model, reg_scaler, reg_columns, reg_background,
 clf_model, clf_scaler, clf_features,
 logistic_model, logistic_scaler, logistic_features,
 stack_model) = assets

# -----------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("🎓 Student Success Pipeline")
st.sidebar.markdown("---")

# The Logical Flow requested by the reviewer
menu_options = [
    "🏠 System Dashboard",
    "Phase 1: Performance Stratification",
    "Phase 2: Predictive Forecasting",
    "Phase 3: Risk Detection",
    "Phase 4: Prescriptive Guidance",
    "Phase 5: Resilience Analysis"
]

page = st.sidebar.radio("Navigate Research Phases:", menu_options)

st.sidebar.markdown("---")
st.sidebar.caption("Institutional Research Tool v2.0")
st.sidebar.write("Explainable AI (XAI) Integrated")

# -----------------------------
# 4. PAGE ROUTING
# -----------------------------

# --- HOME DASHBOARD ---
if page == "🏠 System Dashboard":
    st.title("Student Performance Intelligence System")
    st.subheader("An Integrated Machine Learning & Explainable AI Framework")
    
    st.markdown("""
    ### Project Flow Narrative
    To address academic performance comprehensively, this system operates through a five-stage research pipeline:
    
    1. **Diagnostic Stratification:** Groups students into initial performance tiers.
    2. **Score Forecasting:** Uses Regression and **SHAP** to predict exact exam outcomes.
    3. **Risk Identification:** Utilizes Random Forest ensembles to detect students likely to fail.
    4. **Intervention Logic:** Provides **Counterfactual** suggestions to flip "At-Risk" statuses to "Safe."
    5. **Empirical Validation:** Analyzes the dataset for "Resilient" students to validate model insights.
    """)
    
    st.info("💡 **Getting Started:** Select **Phase 1** from the sidebar to begin a student evaluation.")

# --- PHASE 1: LOGISTIC REGRESSION ---
elif page == "Phase 1: Performance Stratification":
    show_phase1_stratification(logistic_model, logistic_scaler, logistic_features)

# --- PHASE 2: LINEAR REGRESSION + SHAP ---
elif page == "Phase 2: Predictive Forecasting":
    show_phase2_prediction(reg_model, reg_scaler, reg_columns, reg_background)

# --- PHASE 3: RANDOM FOREST RISK ---
elif page == "Phase 3: Risk Detection":
    show_phase3_risk_detection(clf_model, clf_scaler, clf_features)

# --- PHASE 4: COUxNTERFACTUALS ---
elif page == "Phase 4: Prescriptive Guidance":
    show_phase4_counterfactuals(clf_model, clf_scaler, clf_features)

# --- PHASE 5: STACKING MODEL + ANALYSIS ---
elif page == "Phase 5: Resilience Analysis":
    show_phase5_resilience_analysis(stack_model, reg_scaler, reg_columns)