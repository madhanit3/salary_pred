import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

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
# LIGHTWEIGHT CSS
# ========================
st.markdown("""
<style>
    /* Main container */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    
    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8b5cf6, #a78bfa, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Cards */
    .premium-card {
        background: #1e293b;
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid #334155;
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
    
    .result-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .result-amount {
        font-size: 3rem;
        font-weight: 700;
        color: #a78bfa;
        margin: 0.5rem 0;
    }
    
    .result-meta {
        color: #64748b;
        font-size: 0.8rem;
    }
    
    /* Sidebar */
    .sidebar-title {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.1rem;
    }
    
    .sidebar-subtitle {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-box {
        background: #1e293b;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border: 1px solid #334155;
        margin-bottom: 0.6rem;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #f1f5f9;
        font-size: 1.1rem;
        font-weight: 600;
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
        font-size: 0.75rem;
        padding: 2rem 0 0.5rem 0;
        border-top: 1px solid #1e293b;
        margin-top: 2rem;
    }
    
    /* Success animation */
    .success-check {
        font-size: 3rem;
        display: inline-block;
        animation: popIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    @keyframes popIn {
        0% { transform: scale(0); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    /* Feature chips */
    .feature-chip {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        color: #94a3b8;
        font-size: 0.7rem;
        margin: 0.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# SESSION STATE
# ========================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_error" not in st.session_state:
    st.session_state.prediction_error = None
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False

# ========================
# CACHED LOADERS
# ========================
@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    model_path = Path("model.pkl")
    if not model_path.exists():
        st.error("❌ **Model Not Found**\n\nPlease ensure `model.pkl` is in the application directory.")
        st.stop()
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ **Failed to Load Model**\n\n{str(e)}")
        st.stop()

@st.cache_data
def load_dataset():
    """Load the dataset for dropdown values."""
    data_path = Path("Salary Data.csv")
    if not data_path.exists():
        st.error("❌ **Dataset Not Found**\n\nPlease ensure `Salary Data.csv` is in the application directory.")
        st.stop()
    
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            st.error("❌ **Dataset is Empty**\n\nThe CSV file contains no data.")
            st.stop()
        return df
    except Exception as e:
        st.error(f"❌ **Failed to Load Dataset**\n\n{str(e)}")
        st.stop()

# ========================
# HELPER FUNCTIONS
# ========================
def get_unique_values(df: pd.DataFrame, column: str):
    """Get unique values from a column, handling missing values."""
    values = df[column].dropna().unique().tolist()
    return sorted(values)

def format_currency(amount: float) -> str:
    """Format currency in Indian Rupees."""
    if amount >= 10000000:  # Crore
        return f"₹ {amount/10000000:.2f} Cr"
    elif amount >= 100000:  # Lakh
        return f"₹ {amount/100000:.2f} Lakh"
    else:
        return f"₹ {amount:,.2f}"

# ========================
# SIDEBAR
# ========================
def render_sidebar(df: pd.DataFrame):
    """Render the sidebar with project information."""
    with st.sidebar:
        st.markdown('<div class="sidebar-title">💰 Salary Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">AI-Powered Salary Predictor</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset overview
        st.markdown("### 📊 Dataset")
        
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Samples</div>
            <div class="metric-value">{df.shape[0]:,}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Total Features</div>
            <div class="metric-value">{df.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature information
        st.markdown("### 🔍 Features")
        
        # Count numeric and categorical
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Numeric Features</div>
            <div class="metric-value">{len(numeric_cols)}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Categorical Features</div>
            <div class="metric-value">{len(cat_cols)}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature chips
        st.markdown("#### Feature List")
        chip_html = ""
        for col in df.columns[:10]:
            chip_html += f'<span class="feature-chip">{col}</span> '
        if len(df.columns) > 10:
            chip_html += f'<span class="feature-chip">+{len(df.columns)-10} more</span>'
        st.markdown(chip_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model status
        st.markdown("### 🧠 Model")
        st.success("✅ Loaded Successfully")
        st.caption("Pipeline: OneHotEncoder + RandomForestRegressor")

# ========================
# MAIN APP
# ========================
def main():
    """Main application entry point."""
    
    # Load data and model
    df = load_dataset()
    model = load_model()
    
    # Render sidebar
    render_sidebar(df)
    
    # Hero section
    st.markdown("""
    <div class="main-header">
        <div class="hero-title">Salary Predictor</div>
        <div class="hero-subtitle">Machine Learning powered salary estimation</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Input form
    with st.container():
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Enter Employee Details")
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        
        # Create form
        with st.form(key="prediction_form"):
            # Two columns for inputs
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input(
                    "🎂 Age",
                    min_value=18,
                    max_value=80,
                    value=30,
                    step=1,
                    help="Employee's age (18-80)"
                )
                
                gender = st.selectbox(
                    "👤 Gender",
                    options=get_unique_values(df, "Gender"),
                    help="Select employee's gender"
                )
                
                education_level = st.selectbox(
                    "🎓 Education Level",
                    options=get_unique_values(df, "Education Level"),
                    help="Select highest education level"
                )
            
            with col2:
                job_title = st.selectbox(
                    "👔 Job Title",
                    options=get_unique_values(df, "Job Title"),
                    help="Select employee's job title"
                )
                
                years_experience = st.number_input(
                    "💼 Years of Experience",
                    min_value=0,
                    max_value=50,
                    value=5,
                    step=1,
                    help="Years of professional experience (0-50)"
                )
            
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Predict Salary",
                use_container_width=True,
                type="primary"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction placeholder
    prediction_container = st.container()
    
    # Handle form submission
    if submitted:
        try:
            # Create input DataFrame with exact column names and order
            input_data = {
                "Age": [age],
                "Gender": [gender],
                "Education Level": [education_level],
                "Job Title": [job_title],
                "Years of Experience": [years_experience]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # Show prediction progress
            with st.spinner("🧠 Computing prediction..."):
                # Make prediction using the loaded model
                prediction = model.predict(input_df)
                pred_value = float(prediction[0])
                
                # Store in session state
                st.session_state.prediction_result = pred_value
                st.session_state.prediction_error = None
                st.session_state.show_prediction = True
                
        except Exception as e:
            # Store error in session state
            st.session_state.prediction_error = str(e)
            st.session_state.prediction_result = None
            st.session_state.show_prediction = True
    
    # Display prediction result or error
    with prediction_container:
        if st.session_state.show_prediction:
            if st.session_state.prediction_result is not None:
                pred_value = st.session_state.prediction_result
                
                # Display result card
                st.markdown(f"""
                <div class="result-card">
                    <div class="success-check">✅</div>
                    <div class="result-label">Predicted Annual Salary</div>
                    <div class="result-amount">{format_currency(pred_value)}</div>
                    <div class="result-meta">Based on 5 features • Pipeline: OneHotEncoder + RandomForest</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show input summary
                with st.expander("📋 View Input Summary"):
                    summary_data = {
                        "Feature": ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"],
                        "Value": [age, gender, education_level, job_title, years_experience]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
            elif st.session_state.prediction_error is not None:
                # Display error
                st.error("❌ **Prediction Failed**")
                st.code(st.session_state.prediction_error, language="python")
                
                st.info("""
                💡 **Troubleshooting:**
                - Ensure `model.pkl` was trained with the same features
                - Check that all input values are valid
                - Verify the model file is not corrupted
                """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        © 2026 Salary Predictor Pro • Powered by Streamlit & scikit-learn
    </div>
    """, unsafe_allow_html=True)

# ========================
# RUN APP
# ========================
if __name__ == "__main__":
    main()
