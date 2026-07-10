import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any

# ========================
# PAGE CONFIGURATION
# ========================
st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CUSTOM CSS - LIGHTWEIGHT & CLOUD COMPATIBLE
# ========================
def load_css() -> str:
    """Return lightweight CSS compatible with Streamlit Cloud."""
    return """
    <style>
        /* Main container padding */
        .main {
            padding: 0rem 1rem;
        }
        
        /* Gradient text for hero */
        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #7c3aed, #8b5cf6, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .hero-subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.2rem;
            margin-top: -0.5rem;
            margin-bottom: 2rem;
        }
        
        /* Card styling */
        .premium-card {
            background: #1e293b;
            border-radius: 16px;
            padding: 2rem;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        
        /* Result card */
        .result-card {
            background: linear-gradient(135deg, #1e293b, #2d1b69);
            border-radius: 16px;
            padding: 2rem;
            border: 2px solid #7c3aed;
            text-align: center;
            margin-top: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(124, 58, 237, 0.2);
        }
        
        .result-label {
            color: #94a3b8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .result-amount {
            font-size: 3.5rem;
            font-weight: 700;
            color: #a78bfa;
            margin: 0.5rem 0;
        }
        
        .result-meta {
            color: #64748b;
            font-size: 0.85rem;
        }
        
        /* Sidebar styling */
        .sidebar-title {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #7c3aed, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.2rem;
        }
        
        .sidebar-subtitle {
            color: #94a3b8;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: #1e293b;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #334155;
            margin-bottom: 0.75rem;
        }
        
        .metric-label {
            color: #94a3b8;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-value {
            color: #f1f5f9;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        /* Feature chip */
        .feature-chip {
            display: inline-block;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 20px;
            padding: 0.2rem 0.8rem;
            color: #94a3b8;
            font-size: 0.75rem;
            margin: 0.15rem;
        }
        
        /* Divider */
        .custom-divider {
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, #334155, transparent);
            margin: 1.5rem 0;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #475569;
            font-size: 0.8rem;
            padding: 2rem 0 0.5rem 0;
            border-top: 1px solid #1e293b;
            margin-top: 2rem;
        }
        
        /* Success check animation */
        .success-check {
            display: inline-block;
            font-size: 3rem;
            animation: pulse 1s ease;
        }
        
        @keyframes pulse {
            0% { transform: scale(0); }
            60% { transform: scale(1.3); }
            100% { transform: scale(1); }
        }
    </style>
    """

# Load CSS once
if "css_loaded" not in st.session_state:
    st.markdown(load_css(), unsafe_allow_html=True)
    st.session_state.css_loaded = True

# ========================
# INITIALIZE SESSION STATE
# ========================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False

# ========================
# CACHED DATA LOADERS
# ========================
@st.cache_resource
def load_model():
    """Load and cache the machine learning model."""
    model_path = Path("model.pkl")
    if not model_path.exists():
        st.error("❌ **Model Not Found**\n\nPlease ensure `model.pkl` is in the same directory as `app.py`.")
        st.stop()
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ **Failed to Load Model**\n\nError: {str(e)}")
        st.stop()

@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load and cache the salary dataset."""
    data_path = Path("Salary Data.csv")
    if not data_path.exists():
        st.error("❌ **Dataset Not Found**\n\nPlease ensure `Salary Data.csv` is in the same directory as `app.py`.")
        st.stop()
    
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            st.error("❌ **Dataset is Empty**\n\nThe CSV file contains no data.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"❌ **Failed to Load Dataset**\n\nError: {str(e)}")
        st.stop()

# ========================
# HELPER FUNCTIONS
# ========================
def detect_target_column(df: pd.DataFrame) -> str:
    """Detect the target column by checking common names."""
    target_keywords = ["salary", "target", "prediction", "output", "wage", "income"]
    
    for col in df.columns:
        col_lower = col.lower().strip()
        for keyword in target_keywords:
            if keyword in col_lower:
                return col
    
    # If no match, assume the last column is the target
    return df.columns[-1]

def get_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Get all feature columns excluding the target."""
    return [col for col in df.columns if col != target_col]

def format_currency(amount: float) -> str:
    """Format amount as Indian Rupee currency."""
    if amount >= 10000000:  # Crore
        return f"₹ {amount/10000000:.1f} Cr"
    elif amount >= 100000:  # Lakh
        return f"₹ {amount/100000:.1f} Lakh"
    else:
        return f"₹ {amount:,.2f}"

# ========================
# SIDEBAR
# ========================
def render_sidebar(df: pd.DataFrame, feature_cols: List[str], target_col: str):
    """Render the sidebar with project information."""
    with st.sidebar:
        st.markdown('<div class="sidebar-title">💰 Salary Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">AI-Powered Predictor</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset overview
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Samples</div>
                <div class="metric-value">{df.shape[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Features</div>
                <div class="metric-value">{len(feature_cols)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Target Variable</div>
            <div class="metric-value" style="font-size:0.95rem;">{target_col}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature distribution
        st.markdown("### 🔍 Features")
        numeric_count = len([c for c in feature_cols if c in df.select_dtypes(include=np.number).columns])
        categorical_count = len(feature_cols) - numeric_count
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Numeric Features</div>
            <div class="metric-value">{numeric_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Categorical Features</div>
            <div class="metric-value">{categorical_count}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature chips
        st.markdown("#### Feature List")
        chip_html = ""
        for feat in feature_cols[:8]:
            chip_html += f'<span class="feature-chip">{feat}</span> '
        if len(feature_cols) > 8:
            chip_html += f'<span class="feature-chip">+{len(feature_cols)-8} more</span>'
        st.markdown(chip_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model status
        st.markdown("### 🧠 Model Status")
        st.success("✅ Model Loaded Successfully")
        st.caption("model.pkl (scikit-learn Pipeline)")

# ========================
# MAIN APPLICATION
# ========================
def main():
    """Main application entry point."""
    
    # Load data and model
    df = load_dataset()
    model = load_model()
    
    # Detect target and features
    target_col = detect_target_column(df)
    feature_cols = get_feature_columns(df, target_col)
    
    # Render sidebar
    render_sidebar(df, feature_cols, target_col)
    
    # Main content - Hero Section
    st.markdown('<div class="hero-title">Salary Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Intelligent salary estimation powered by machine learning</div>', unsafe_allow_html=True)
    
    # Input Section
    with st.container():
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Enter Employee Details")
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        
        # Create form
        with st.form(key="prediction_form"):
            # Input widgets
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input(
                    "🎂 Age",
                    min_value=18,
                    max_value=80,
                    value=30,
                    step=1,
                    help="Enter the employee's age (18-80)"
                )
                
                gender = st.selectbox(
                    "👤 Gender",
                    options=["Male", "Female", "Other"],
                    help="Select the employee's gender"
                )
                
                education_level = st.selectbox(
                    "🎓 Education Level",
                    options=["Bachelor's", "Master's", "PhD", "High School", "Associate"],
                    help="Select the highest education level"
                )
            
            with col2:
                job_title = st.selectbox(
                    "👔 Job Title",
                    options=["Software Engineer", "Data Scientist", "Product Manager", 
                            "Business Analyst", "DevOps Engineer", "UX Designer",
                            "Project Manager", "Data Analyst", "ML Engineer"],
                    help="Select the employee's job title"
                )
                
                years_experience = st.number_input(
                    "💼 Years of Experience",
                    min_value=0,
                    max_value=50,
                    value=5,
                    step=1,
                    help="Enter years of professional experience (0-50)"
                )
            
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Predict Salary",
                use_container_width=True,
                type="primary"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction placeholder - fixed position
    prediction_placeholder = st.container()
    
    # Prediction logic
    if submitted:
        try:
            # Create input DataFrame exactly as required
            input_data = {
                'Age': [age],
                'Gender': [gender],
                'Education Level': [education_level],
                'Job Title': [job_title],
                'Years of Experience': [years_experience]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # Show processing
            with st.spinner("🧠 Analyzing features and computing prediction..."):
                # Make prediction
                prediction = model.predict(input_df)
                pred_value = float(prediction[0])
                
                # Store in session state
                st.session_state.prediction_result = pred_value
                st.session_state.prediction_error = None
                st.session_state.show_prediction = True
                
        except Exception as e:
            st.session_state.prediction_error = str(e)
            st.session_state.prediction_result = None
            st.session_state.show_prediction = True
    
    # Display prediction result
    with prediction_placeholder:
        if st.session_state.show_prediction:
            if st.session_state.prediction_result is not None:
                pred_value = st.session_state.prediction_result
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="success-check">✓</div>
                    <div class="result-label">Predicted Annual Salary</div>
                    <div class="result-amount">{format_currency(pred_value)}</div>
                    <div class="result-meta">Based on 5 features • Model: RandomForest Pipeline</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show input summary
                with st.expander("📋 View Input Summary", expanded=False):
                    summary_data = {
                        "Feature": ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"],
                        "Value": [age, gender, education_level, job_title, years_experience]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
            elif st.session_state.prediction_error is not None:
                st.error(f"❌ **Prediction Failed**\n\n{st.session_state.prediction_error}")
                
                st.info("💡 **Troubleshooting Tips:**\n"
                       "- Ensure `model.pkl` was trained with the same features\n"
                       "- Check that all input values are valid\n"
                       "- Verify the model file is not corrupted")
    
    # Footer
    st.markdown("""
    <div class="footer">
        © 2026 Salary Predictor Pro • Built with Streamlit & scikit-learn
    </div>
    """, unsafe_allow_html=True)

# ========================
# APPLICATION ENTRY POINT
# ========================
if __name__ == "__main__":
    main()
