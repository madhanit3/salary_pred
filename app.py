import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any
import base64

# ========================
# PAGE CONFIGURATION
# ========================
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================
# CUSTOM CSS - PREMIUM SAAS UI
# ========================
def load_css() -> str:
    """Return premium CSS for the application."""
    return """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Reset and Base */
        .stApp {
            background: linear-gradient(135deg, #0c0e1a 0%, #1a1f3a 30%, #2d1b69 70%, #1a1f3a 100%);
            background-attachment: fixed;
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        .stDeployButton {display: none;}
        footer {visibility: hidden;}
        
        /* Main Container */
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem 1.5rem;
        }
        
        /* Glassmorphism Card */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            padding: 2.5rem 2.5rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.15);
            box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.6);
            transform: translateY(-2px);
        }
        
        .glass-card::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.03) 0%, transparent 70%);
            pointer-events: none;
        }
        
        /* Hero Section */
        .hero-section {
            text-align: center;
            padding: 2rem 0 2.5rem 0;
            position: relative;
        }
        
        .hero-title {
            font-size: 4.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #ffecd2 100%);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 4s ease-in-out infinite;
            margin-bottom: 0.5rem;
            letter-spacing: -1px;
        }
        
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .hero-subtitle {
            color: rgba(255, 255, 255, 0.6);
            font-size: 1.15rem;
            font-weight: 300;
            letter-spacing: 1px;
            margin-top: -0.5rem;
        }
        
        /* Input Labels */
        .input-label {
            color: rgba(255, 255, 255, 0.8);
            font-weight: 500;
            font-size: 0.85rem;
            letter-spacing: 0.3px;
            margin-bottom: 0.3rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .input-icon {
            font-size: 1.1rem;
        }
        
        /* Custom Input Styling */
        .stNumberInput > div, .stSelectbox > div {
            background: rgba(255, 255, 255, 0.06) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            transition: all 0.3s ease !important;
        }
        
        .stNumberInput > div:hover, .stSelectbox > div:hover {
            border-color: rgba(255, 255, 255, 0.2) !important;
            background: rgba(255, 255, 255, 0.08) !important;
        }
        
        .stNumberInput > div:focus-within, .stSelectbox > div:focus-within {
            border-color: #f5576c !important;
            box-shadow: 0 0 0 3px rgba(245, 87, 108, 0.15) !important;
        }
        
        /* Predict Button */
        .predict-btn {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 60px !important;
            padding: 0.8rem 3rem !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            letter-spacing: 0.5px;
            box-shadow: 0 15px 35px -8px rgba(245, 87, 108, 0.4) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100%;
            position: relative;
            overflow: hidden;
        }
        
        .predict-btn:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 20px 45px -8px rgba(245, 87, 108, 0.6) !important;
        }
        
        .predict-btn:active {
            transform: scale(0.98);
        }
        
        /* Prediction Result Card */
        .result-card {
            background: linear-gradient(135deg, rgba(255, 215, 0, 0.08), rgba(245, 87, 108, 0.08));
            border: 2px solid rgba(255, 215, 0, 0.15);
            border-radius: 30px;
            padding: 2.5rem 2rem;
            text-align: center;
            animation: fadeInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .result-card::before {
            content: '✨';
            position: absolute;
            top: -20px;
            right: -20px;
            font-size: 6rem;
            opacity: 0.05;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        .result-label {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 3px;
            font-weight: 500;
        }
        
        .result-amount {
            font-size: 4.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 50%, #f5576c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0.5rem 0;
            letter-spacing: -2px;
        }
        
        .result-meta {
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }
        
        /* Success Animation */
        .success-check {
            display: inline-block;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4CAF50, #45a049);
            line-height: 60px;
            text-align: center;
            font-size: 2rem;
            color: white;
            animation: scaleIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            margin-bottom: 1rem;
        }
        
        @keyframes scaleIn {
            0% { transform: scale(0); }
            60% { transform: scale(1.2); }
            100% { transform: scale(1); }
        }
        
        /* Sidebar */
        .css-1d391kg, .css-12oz5g7 {
            background: rgba(12, 14, 26, 0.85) !important;
            backdrop-filter: blur(20px) saturate(180%);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .sidebar-title {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #f093fb, #f5576c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        
        .sidebar-subtitle {
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.8rem;
            letter-spacing: 1px;
            margin-bottom: 2rem;
        }
        
        .sidebar-metric {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }
        
        .sidebar-metric-label {
            color: rgba(255, 255, 255, 0.4);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .sidebar-metric-value {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        /* Progress Bar */
        .progress-container {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 0.5rem;
            margin: 1rem 0;
        }
        
        .progress-bar {
            height: 6px;
            background: linear-gradient(90deg, #f093fb, #f5576c);
            border-radius: 20px;
            transition: width 1s ease;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem 0 0.5rem 0;
            color: rgba(255, 255, 255, 0.15);
            font-size: 0.75rem;
            letter-spacing: 1px;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2.5rem;
            }
            .result-amount {
                font-size: 2.8rem;
            }
            .glass-card {
                padding: 1.5rem 1rem;
            }
            .main-container {
                padding: 1rem 0.8rem;
            }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #f093fb, #f5576c);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #f5576c, #f093fb);
        }
        
        /* Spinner Animation */
        .stSpinner > div {
            border-color: #f5576c transparent #f5576c transparent !important;
            border-width: 4px !important;
            width: 50px !important;
            height: 50px !important;
        }
        
        /* Divider */
        .custom-divider {
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
            height: 1px;
            border: none;
            margin: 2rem 0;
        }
        
        /* Feature Chip */
        .feature-chip {
            display: inline-block;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 20px;
            padding: 0.2rem 0.8rem;
            color: rgba(255,255,255,0.6);
            font-size: 0.7rem;
            margin: 0.15rem;
        }
    </style>
    """

#st.markdown(load_css(), unsafe_allow_html=True)

# ========================
# CACHED DATA LOADERS
# ========================
@st.cache_resource
def load_model() -> Any:
    """Load and cache the machine learning model from model.pkl using joblib."""
    model_path = Path("model.pkl")
    if not model_path.exists():
        st.error("❌ **Model Not Found**\n\nPlease ensure `model.pkl` is in the same directory as `app.py`.")
        st.stop()
    
    try:
        # Try loading with joblib first (preferred for scikit-learn models)
        try:
            model = joblib.load(model_path)
        except:
            # Fallback to pickle if joblib fails
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ **Failed to Load Model**\n\nError: {str(e)}")
        st.stop()

@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load and cache the salary dataset from Salary Data.csv."""
    data_path = Path("Salary Data.csv")
    if not data_path.exists():
        st.error("❌ **Dataset Not Found**\n\nPlease ensure `Salary Data.csv` is in the same directory as `app.py`.")
        st.stop()
    
    try:
        df = pd.read_csv(data_path)
        if df.empty:
            st.error("❌ **Dataset is Empty**\n\nThe CSV file contains no data. Please check the file.")
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

def categorize_columns(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """Separate features into numeric and categorical."""
    feature_cols = [col for col in df.columns if col != target_col]
    
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numeric_cols, categorical_cols

def create_input_widgets(
    numeric_cols: List[str],
    categorical_cols: List[str],
    df: pd.DataFrame
) -> Dict[str, Any]:
    """Create Streamlit input widgets for all features."""
    inputs = {}
    
    # Create columns for better layout
    cols = st.columns(2)
    
    all_features = numeric_cols + categorical_cols
    mid = len(all_features) // 2 + len(all_features) % 2
    left_features = all_features[:mid]
    right_features = all_features[mid:]
    
    # Icons for different features
    icons = {
        'age': '🎂',
        'experience': '💼',
        'education': '🎓',
        'gender': '👤',
        'location': '📍',
        'department': '🏢',
        'job_title': '👔',
        'years': '📅',
        'level': '📊',
        'rating': '⭐',
        'score': '🏆'
    }
    
    # Left column
    with cols[0]:
        for col in left_features:
            # Find appropriate icon
            icon = "📌"
            for key, value in icons.items():
                if key in col.lower():
                    icon = value
                    break
            
            if col in numeric_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                step = 0.1 if (max_val - min_val) < 10 else 1.0
                
                st.markdown(f'<div class="input-label"><span class="input-icon">{icon}</span> {col}</div>', unsafe_allow_html=True)
                inputs[col] = st.number_input(
                    label=col,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=step,
                    format="%.2f" if step < 1 else "%.0f",
                    label_visibility="collapsed",
                    key=f"num_{col}"
                )
            else:
                unique_values = sorted(df[col].dropna().unique().tolist())
                st.markdown(f'<div class="input-label"><span class="input-icon">{icon}</span> {col}</div>', unsafe_allow_html=True)
                inputs[col] = st.selectbox(
                    label=col,
                    options=unique_values,
                    label_visibility="collapsed",
                    key=f"cat_{col}"
                )
    
    # Right column
    with cols[1]:
        for col in right_features:
            icon = "📌"
            for key, value in icons.items():
                if key in col.lower():
                    icon = value
                    break
            
            if col in numeric_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                step = 0.1 if (max_val - min_val) < 10 else 1.0
                
                st.markdown(f'<div class="input-label"><span class="input-icon">{icon}</span> {col}</div>', unsafe_allow_html=True)
                inputs[col] = st.number_input(
                    label=col,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=step,
                    format="%.2f" if step < 1 else "%.0f",
                    label_visibility="collapsed",
                    key=f"num_{col}"
                )
            else:
                unique_values = sorted(df[col].dropna().unique().tolist())
                st.markdown(f'<div class="input-label"><span class="input-icon">{icon}</span> {col}</div>', unsafe_allow_html=True)
                inputs[col] = st.selectbox(
                    label=col,
                    options=unique_values,
                    label_visibility="collapsed",
                    key=f"cat_{col}"
                )
    
    return inputs

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
    """Render the premium sidebar with project information."""
    with st.sidebar:
        st.markdown('<div class="sidebar-title">💰 Salary Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">AI-Powered Salary Predictor</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Dataset metrics
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="sidebar-metric">
                <div class="sidebar-metric-label">Samples</div>
                <div class="sidebar-metric-value">{df.shape[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="sidebar-metric">
                <div class="sidebar-metric-label">Features</div>
                <div class="sidebar-metric-value">{len(feature_cols)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Target Variable</div>
            <div class="sidebar-metric-value" style="font-size:0.95rem;">{target_col}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature distribution
        st.markdown("### 🔍 Features")
        numeric_count = len([c for c in feature_cols if c in df.select_dtypes(include=np.number).columns])
        categorical_count = len(feature_cols) - numeric_count
        
        st.markdown(f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Numeric Features</div>
            <div class="sidebar-metric-value">{numeric_count}</div>
        </div>
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">Categorical Features</div>
            <div class="sidebar-metric-value">{categorical_count}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature chips
        st.markdown("#### Feature List")
        chip_html = ""
        for feat in feature_cols[:8]:  # Show first 8
            chip_html += f'<span class="feature-chip">{feat}</span> '
        if len(feature_cols) > 8:
            chip_html += f'<span class="feature-chip">+{len(feature_cols)-8} more</span>'
        st.markdown(chip_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 🧠 Model Status")
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; color: #4CAF50;">
            <span style="font-size:1.2rem;">●</span> <span>Loaded Successfully</span>
        </div>
        <div style="color:rgba(255,255,255,0.3); font-size:0.7rem; margin-top:4px;">
            model.pkl (loaded with joblib)
        </div>
        """, unsafe_allow_html=True)

# ========================
# MAIN APPLICATION
# ========================
def main():
    """Main application entry point."""
    
    # Load data and model
    with st.spinner("🔄 Loading application resources..."):
        try:
            df = load_dataset()
            model = load_model()
        except Exception as e:
            st.error(f"Failed to initialize application: {str(e)}")
            st.stop()
    
    # Detect target column
    target_col = detect_target_column(df)
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Categorize columns
    numeric_cols, categorical_cols = categorize_columns(df, target_col)
    
    # Render sidebar
    render_sidebar(df, feature_cols, target_col)
    
    # Main content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Salary Predictor</div>
        <div class="hero-subtitle">Intelligent salary estimation powered by machine learning</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Enter Employee Details")
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        
        # Create input widgets
        inputs = create_input_widgets(numeric_cols, categorical_cols, df)
        
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        
        # Predict button
        predict_clicked = st.button(
            "🚀 Predict Salary",
            use_container_width=True,
            type="primary"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction Logic
    if predict_clicked:
        try:
            # Build input DataFrame
            input_df = pd.DataFrame([inputs])
            
            # Ensure correct column order
            input_df = input_df[feature_cols]
            
            # Show prediction progress
            with st.spinner("🧠 Analyzing features and computing prediction..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress_bar.progress(i + 1)
                time.sleep(0.2)
                progress_bar.empty()
            
            # Make prediction
            prediction = model.predict(input_df)
            pred_value = float(prediction[0])
            
            # Display result
            st.markdown("""
            <div class="result-card">
                <div class="success-check">✓</div>
                <div class="result-label">Predicted Annual Salary</div>
                <div class="result-amount">{}</div>
                <div class="result-meta">Based on {} features • Model: model.pkl</div>
            </div>
            """.format(format_currency(pred_value), len(feature_cols)), unsafe_allow_html=True)
            
            # Show input summary in expander
            with st.expander("📋 View Input Summary"):
                st.dataframe(input_df, use_container_width=True)
                
                # Show feature importance or details
                st.markdown("#### Feature Breakdown")
                for col in feature_cols:
                    st.markdown(f"**{col}:** {inputs[col]}")
        
        except KeyError as e:
            st.error(f"❌ **Feature Mismatch**\n\nMissing feature: {str(e)}\n\nPlease ensure all required features are provided.")
        
        except Exception as e:
            st.error(f"❌ **Prediction Failed**\n\n{str(e)}")
            
            # Provide helpful suggestions
            st.info("💡 **Troubleshooting Tips:**\n"
                   "- Ensure `model.pkl` was trained with the same features as in `Salary Data.csv`\n"
                   "- Check that all input values are valid\n"
                   "- Verify the model file is not corrupted")
    
    # Footer
    st.markdown("""
    <div class="footer">
        © 2026 Salary Predictor Pro • Powered by Streamlit & Machine Learning
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========================
# APPLICATION ENTRY POINT
# ========================
if __name__ == "__main__":
    main()
