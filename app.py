import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# ========================
# SIMPLE CSS
# ========================
st.markdown("""
<style>
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1e293b, #2d1b69);
        border-radius: 16px;
        padding: 2rem;
        border: 2px solid #7c3aed;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-amount {
        font-size: 3rem;
        font-weight: 700;
        color: #a78bfa;
        margin: 0.5rem 0;
    }
    .result-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .result-meta {
        color: #64748b;
        font-size: 0.8rem;
    }
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        padding: 2rem 0 0.5rem 0;
        border-top: 1px solid #1e293b;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# SESSION STATE
# ========================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False

# ========================
# CACHED LOADERS
# ========================
@st.cache_resource
def load_model():
    """Load the trained model."""
    if not Path("model.pkl").exists():
        st.error("❌ model.pkl not found!")
        st.stop()
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

@st.cache_data
def load_dataset():
    """Load the dataset."""
    if not Path("Salary Data.csv").exists():
        st.error("❌ Salary Data.csv not found!")
        st.stop()
    try:
        df = pd.read_csv("Salary Data.csv")
        if df.empty:
            st.error("❌ Dataset is empty!")
            st.stop()
        return df
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        st.stop()

# ========================
# HELPER FUNCTIONS
# ========================
def get_unique_values(df, column):
    """Get unique values from a column."""
    return sorted(df[column].dropna().unique().tolist())

def format_currency(amount):
    """Format currency."""
    if amount >= 10000000:
        return f"₹ {amount/10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹ {amount/100000:.2f} Lakh"
    else:
        return f"₹ {amount:,.2f}"

# ========================
# MAIN APP
# ========================
def main():
    # Load data and model
    df = load_dataset()
    model = load_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 💰 Salary Pro")
        st.markdown("---")
        st.metric("Total Samples", f"{df.shape[0]:,}")
        st.metric("Total Features", df.shape[1])
        st.markdown("---")
        st.success("✅ Model Loaded")
        st.caption("Pipeline: OneHotEncoder + RandomForest")
    
    # Hero Section
    st.markdown('<div class="hero-title">Salary Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Machine Learning powered salary estimation</div>', unsafe_allow_html=True)
    
    # Input Form
    with st.form(key="prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
            gender = st.selectbox("Gender", options=get_unique_values(df, "Gender"))
            education = st.selectbox("Education Level", options=get_unique_values(df, "Education Level"))
        
        with col2:
            job_title = st.selectbox("Job Title", options=get_unique_values(df, "Job Title"))
            experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, step=1)
        
        submitted = st.form_submit_button("🚀 Predict Salary", use_container_width=True, type="primary")
    
    # Prediction Logic
    if submitted:
        try:
            input_df = pd.DataFrame({
                "Age": [age],
                "Gender": [gender],
                "Education Level": [education],
                "Job Title": [job_title],
                "Years of Experience": [experience]
            })
            
            with st.spinner("Computing prediction..."):
                prediction = model.predict(input_df)
                st.session_state.prediction_result = float(prediction[0])
                st.session_state.show_prediction = True
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.session_state.show_prediction = False
    
    # Display Result
    if st.session_state.show_prediction and st.session_state.prediction_result is not None:
        pred_value = st.session_state.prediction_result
        
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:3rem;">✅</div>
            <div class="result-label">Predicted Annual Salary</div>
            <div class="result-amount">{format_currency(pred_value)}</div>
            <div class="result-meta">Based on 5 features • RandomForest Model</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show input summary
        with st.expander("📋 View Input Summary"):
            summary_df = pd.DataFrame({
                "Feature": ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"],
                "Value": [age, gender, education, job_title, experience]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        © 2026 Salary Predictor • Powered by Streamlit & scikit-learn
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
