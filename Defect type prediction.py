import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import plotly.express as px
import plotly.graph_objects as graph_objects

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="PRISM AM | LPBF Parameter Optimization Suite",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Dark-Theme Adjustments & Styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B, #FF8F00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #A3A8B4;
        font-style: italic;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1.05rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# APP BRANDING LOGO (INLINE VECTOR DESIGN)
# ==============================================================================
logo_svg = """
<svg xmlns="http://w3.org" viewBox="0 0 500 100" width="100%" height="80">
  <defs>
    <linearGradient id="laserGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#FF4B4B" />
      <stop offset="100%" stop-color="#FF8F00" />
    </linearGradient>
  </defs>
  <!-- Geometric Laser Melting Nozzle Icon -->
  <path d="M25 15 L55 15 L45 50 L35 50 Z" fill="none" stroke="url(#laserGrad)" stroke-width="3" stroke-linejoin="round"/>
  <path d="M40 50 L40 75" fill="none" stroke="#FF4B4B" stroke-width="2" stroke-dasharray="4,3"/>
  <ellipse cx="40" cy="78" rx="12" ry="4" fill="none" stroke="#FF8F00" stroke-width="2"/>
  <polygon points="40,74 46,78 40,82 34,78" fill="#FF4B4B" opacity="0.7"/>
  <!-- Typography -->
  <text x="85" y="52" font-family="'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-weight="900" font-size="34" fill="#FFFFFF" letter-spacing="2">PRISM <tspan fill="url(#laserGrad)">AM</tspan></text>
  <text x="87" y="74" font-family="'Segoe UI', Roboto, Helvetica, Arial, sans-serif" font-weight="500" font-size="13" fill="#8A93A6" letter-spacing="1.5">PREDICTIVE PARAMETER OPTIMIZER</text>
</svg>
"""

# Render Branding Header
st.sidebar.markdown(logo_svg, unsafe_allow_html=True)
st.sidebar.write("---")

# ==============================================================================
# CONSTANTS & CONFIGURATIONS
# ==============================================================================
INPUT_COLS = ["Power", "Speed", "ExposureTime", "EnergyDensity", "Al", "Fe", "Cr", "Ti", "Si"]
REG_COLS = ["Depth", "Width"]
CLS_COL = "DefectType"
REQUIRED_COLS = INPUT_COLS + REG_COLS + [CLS_COL]

# Initialize Missing Session States Cleanly
for state_key in ["model", "scaler", "le", "df", "history", "dataset_source"]:
    if state_key not in st.session_state:
        st.session_state[state_key] = None

# Main Page Title Layout
st.markdown('<div class="main-title">Laser Powder Bed Fusion Optimization Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid Finite Element Method (FEM) + Multi-Task Deep Learning Framework</div>', unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR CONTROL INTERFACES
# ==============================================================================
st.sidebar.header("⚙️ Model Architecture Setup")
hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 3, help="Deeper networks capture complex structural thermal histories.")
neurons_per_layer = st.sidebar.slider("Neurons per Layer", 16, 128, 64, 8, help="Width of internal representations.")
epochs = st.sidebar.slider("Training Steps (Epochs)", 10, 300, 120, 10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)

# ==============================================================================
# CORE NAVIGATION TABS
# ==============================================================================
tabs = st.tabs([
    "📋 System Overview & Architecture", 
    "📊 Training Engine & Data Feed", 
    "🔮 Multi-Objective Real-Time Inference"
])

# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW & METALLURGICAL METHODOLOGY
# ------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("💡 Digital Twin Methodology Bridge")
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        st.markdown("""
        Physical trial-and-error modeling on LPBF machinery is commercially unsustainable. 
        This engine utilizes **Transfer Learning** to systematically bridge high-fidelity physical simulations with rapid real-world execution profiles:
        
        * **Phase I (Source Engine):** The model constructs baseline physical heuristics using raw, macro-scale transient thermal Finite Element Method (FEM) calculation tracks.
        * **Phase II (Target Adaptation):** The framework ingests narrow empirical machine arrays to fine-tune internal multi-variable boundaries.
        
        #### **Target Co-dependencies Modeled Simultaneously:**
        1. **Melt Pool Geometries:** Continual linear estimation of track metric parameters ($\mu m$).
        2. **Solidification Defect Thresholding:** Multiclass classification evaluating safe printing parameters vs. structural flaws.
        """)
        
    with col2:
        st.info("""
        **🚀 Core Operational Blueprint:**
        
        1. **Ingest Empirical Run Matrix:** Transition to the *Training Engine* tab to link your target metallurgical observation database.
        2. **Configure Neural Layer Depths:** Tweak structural weights inside the left configuration panel.
        3. **Map the Physics Window:** Compile data features to execute unified Multi-Task deep network propagation.
        4. **Simulate Boundaries:** Use target inference handles to digitally profile safe melt-track behaviors instantly.
        """)

    st.subheader("📈 Multi-Task Topology Flow diagram")
    st.code("""
     [ Input Features: Laser Power, Scan Velocity, Core Elemental Proportions (Al, Fe, Cr, Ti, Si) ]
                                             │
                                             ▼
                             [ Shared Latent Dense Core Layers ]
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          [ Regression Output Branch ]                [ Categorical Output Branch ]
          Predicts: Melt Depth & Track Width          Predicts: Keyholing / Lack of Fusion / Stable
    """, language="text")

# ------------------------------------------------------------------------------
# TAB 2: TRAINING ENGINE & METRIC COMPILATION
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("📊 Empirical Data Management Hub")
    
    # Showcase Structuring Templates
    with st.expander("🔍 View Expected Database Template Geometry Configuration", expanded=False):
        template_df = pd.DataFrame({
            "Power": [180, 220, 260], "Speed": [800, 1000, 1200], "ExposureTime": [40, 60, 90],
            "EnergyDensity": [55, 65, 80], "Al": [94.5, 90.0, 88.5], "Fe": [1.8, 4.0, 5.5],
            "Cr": [1.4, 2.5, 3.2], "Ti": [1.2, 1.8, 2.0], "Si": [1.1, 1.7, 0.8],
            "Depth": [120.5, 450.2, 980.1], "Width": [230.1, 510.4, 890.7],
            "DefectType": ["Stable-Zone", "LackOfFusion", "Keyholing"]
        })
        st.dataframe(template_df, use_container_width=True, hide_index=True)

    data_option = st.radio("System Input Pipeline Data Target Source:", ["🧪 Use Synthetic Framework Data", "📁 Load Custom Machine Run (CSV)"], horizontal=True)

    # Ingest Data or Compile Synthetic Engine
    if data_option == "🧪 Use Synthetic Framework Data":
        # Safe algorithmic fallback generator to give immediate client utility
        np.random.seed(42)
        rows = 250
        mock_p = np.random.uniform(100, 400, rows)
        mock_v = np.random.uniform(400, 1600, rows)
        mock_ed = (mock_p / (mock_v * 0.1 * 0.05)) * 0.1
        
        mock_depth = (mock_p * 1.8) - (mock_v * 0.15) + np.random.normal(0, 15, rows)
        mock_width = (mock_p * 2.2) - (mock_v * 0.08) + np.random.normal(0, 20, rows)
        
        defect_labels = []
        for p, v in zip(mock_p, mock_v):
            ratio = p / v
            if ratio < 0.15: defect_labels.append("LackOfFusion")
            elif ratio > 0.45: defect_labels.append("Keyholing")
            else: defect_labels.append("Stable-Zone")
            
        synthetic_df = pd.DataFrame({
            "Power": mock_p, "Speed": mock_v, "ExposureTime": np.random.uniform(30, 100, rows),
            "EnergyDensity": mock_ed, "Al": np.random.uniform(85, 95, rows), "Fe": np.random.uniform(1, 5, rows),
            "Cr": np.random.uniform(1, 4, rows), "Ti": np.random.uniform(0.5, 2, rows), "Si": np.random.uniform(0.5, 2, rows),
            "Depth": np.clip(mock_depth, 30, 1200), "Width": np.clip(mock_width, 50, 1500), "DefectType": defect_labels
        })
        st.session_state.df = synthetic_df
        st.session_state.dataset_source = "Synthetic Sandbox Matrix Engine"
        st.success("✅ Sandboxed multi-component alloy data matrix generated securely.")
    else:
