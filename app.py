import streamlit as st
import pandas as pd
import joblib
import os
import math

# Must be the very first Streamlit command
st.set_page_config(
    page_title="ChurnAI Analytics", 
    layout="wide", 
    page_icon="🔮",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Aesthetic
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700&display=swap');

    /* Global Typography */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background & Main Panel */
    .stApp {
        background: radial-gradient(circle at 15% 50%, #0d1117 0%, #000000 100%);
        color: #e6edf3;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(13, 17, 23, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Titles */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: #ffffff;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        padding-bottom: 10px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #8b949e;
        margin-top: 0;
        margin-bottom: 30px;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px 8px 0px 0px;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.05);
        border-bottom: none;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(79, 172, 254, 0.1) !important;
        color: #4facfe !important;
        border-top: 2px solid #4facfe;
    }
    
    /* Inputs */
    .stSelectbox label, .stNumberInput label {
        color: #a5b4fc !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px;
        color: white;
    }
    .stNumberInput input {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px;
        color: white !important;
    }
    
    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #000;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 15px 30px;
        border-radius: 12px;
        border: none;
        width: 100%;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.4);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-top: 20px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(79, 172, 254, 0.6);
        color: #fff;
    }

    /* Result Cards */
    .result-glass {
        background: rgba(13, 17, 23, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.8s ease-out;
    }
    .result-churn {
        box-shadow: 0 10px 40px rgba(255, 75, 75, 0.2);
        border-top: 4px solid #ff4b4b;
    }
    .result-safe {
        box-shadow: 0 10px 40px rgba(0, 255, 136, 0.2);
        border-top: 4px solid #00ff88;
    }
    .prob-metric {
        font-family: 'Outfit', sans-serif;
        font-size: 4rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .prob-churn-text { color: #ff4b4b; }
    .prob-safe-text { color: #00ff88; }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hide top header */
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8633/8633190.png", width=80)
    st.markdown("## ChurnAI Dashboard")
    st.markdown("Advanced predictive analytics powered by ensemble machine learning.")
    st.markdown("---")
    st.markdown("### 🛠️ System Status")
    st.success("🟢 API Online")
    st.info("ℹ️ Model Version: v2.1.0")
    st.markdown("---")
    st.markdown("<span style='color:#8b949e;font-size:0.8rem;'>Built for precision and performance.</span>", unsafe_allow_html=True)

# ----------------- MAIN APP -----------------
st.markdown('<p class="hero-title">Predict Customer Churn</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Input real-time customer data to forecast retention probability with high accuracy.</p>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_models():
    model_path = "model.pkl"
    info_path = "feature_info.pkl"
    if not os.path.exists(model_path) or not os.path.exists(info_path):
        return None, None
    model = joblib.load(model_path)
    feature_info = joblib.load(info_path)
    return model, feature_info

model, feature_info = load_models()

if model is None or feature_info is None:
    st.error("Model files not found! Please run `train.py` first.")
    st.stop()

# Organize features into tabs
features = list(feature_info.keys())
features_per_tab = 12
num_tabs = math.ceil(len(features) / features_per_tab)

tab_names = [f"Data Segment {i+1}" for i in range(num_tabs)]
# Rename some tabs if we want to be fancy
if num_tabs >= 1: tab_names[0] = "👤 Profile & Demographics"
if num_tabs >= 2: tab_names[1] = "📈 Usage & Billing"
if num_tabs >= 3: tab_names[2] = "📱 Devices & Services"
if num_tabs >= 4: tab_names[3] = "⚙️ Additional Metrics"

tabs = st.tabs(tab_names)
input_data = {}

# Distribute features across tabs
for i, tab in enumerate(tabs):
    with tab:
        st.markdown("<br>", unsafe_allow_html=True)
        # Get chunk of features for this tab
        chunk = features[i * features_per_tab : (i + 1) * features_per_tab]
        
        # Create a 3-column layout inside the tab
        cols = st.columns(3, gap="large")
        
        for idx, feature in enumerate(chunk):
            meta = feature_info[feature]
            col = cols[idx % 3]
            
            with col:
                if meta['type'] == 'categorical':
                    options = meta['options']
                    cleaned_options = [opt for opt in options if pd.notna(opt)]
                    
                    # If categorical options are exactly [0, 1] or [1, 0]
                    if set(cleaned_options) == {0, 1} or set(cleaned_options) == {0.0, 1.0}:
                        choice = st.selectbox(
                            label=f"{feature}", 
                            options=["No", "Yes"],
                            help=f"Select Yes or No for {feature}"
                        )
                        input_data[feature] = 1 if choice == "Yes" else 0
                    else:
                        input_data[feature] = st.selectbox(
                            label=f"{feature}", 
                            options=cleaned_options,
                            help=f"Select the value for {feature}"
                        )
                elif meta['type'] == 'numeric':
                    min_val = float(meta.get('min', 0.0))
                    max_val = float(meta.get('max', 10000.0))
                    median_val = float(meta.get('median', 0.0))
                    
                    if min_val == 0.0 and max_val == 1.0:
                        # It's a binary field!
                        choice = st.selectbox(
                            label=f"{feature}",
                            options=["No", "Yes"],
                            index=1 if median_val > 0.5 else 0,
                            help=f"Select Yes or No for {feature}"
                        )
                        input_data[feature] = 1 if choice == "Yes" else 0
                    else:
                        step = 1.0 if (max_val - min_val) > 10 else 0.1
                        
                        input_data[feature] = st.number_input(
                            label=f"{feature}", 
                            min_value=min_val, 
                            max_value=max_val, 
                            value=median_val,
                            step=step,
                            help=f"Range: {min_val} to {max_val}"
                        )
        st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")

# Centered Predict Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔮 GENERATE PREDICTION")

if predict_btn:
    df_input = pd.DataFrame([input_data])
    
    with st.spinner("Analyzing neural pathways and decision trees..."):
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        churn_prob = probabilities[1]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-glass result-churn">
                <h3 style="color: #ff4b4b; text-transform: uppercase; letter-spacing: 2px;">⚠️ High Churn Risk Detected</h3>
                <div class="prob-metric prob-churn-text">{churn_prob:.1%}</div>
                <p style="color: #c9d1d9; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                    This customer's profile aligns with historical patterns of churn. 
                    We recommend immediate proactive intervention by the retention team.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-glass result-safe">
                <h3 style="color: #00ff88; text-transform: uppercase; letter-spacing: 2px;">✅ Stable Customer Profile</h3>
                <div class="prob-metric prob-safe-text">{churn_prob:.1%}</div>
                <p style="color: #c9d1d9; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                    This customer exhibits strong loyalty indicators. No immediate retention action is required.
                </p>
            </div>
            """, unsafe_allow_html=True)
