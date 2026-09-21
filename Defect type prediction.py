import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "logo.png"
)

st.set_page_config(
    page_title="OPTIMUM AM",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# PROFESSIONAL DARK UI
# ==============================================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #06101d 0%, #081827 55%, #06111e 100%);
        color: #eaf4ff;
    }
    [data-testid="stHeader"] { background: rgba(4, 12, 22, 0.92); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071625 0%, #06121f 100%);
        border-right: 1px solid #18324a;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    .brand-card {
        padding: 8px 8px 18px 8px;
        border-bottom: 1px solid #18324a;
        margin-bottom: 18px;
    }
    .brand-name {
        font-size: 20px; font-weight: 800; letter-spacing: .2px;
        color: #f4f8ff; margin-top: 6px;
    }
    .brand-name span { color: #19a9ff; }
    .brand-subtitle { font-size: 11px; color: #91aac0; margin-top: 2px; }
    .main-title {
        font-size: 32px; font-weight: 800; line-height: 1.15;
        margin: 4px 0 4px 0; color: #f3f7ff;
    }
    .main-subtitle {
        font-size: 16px; color: #8fc8ec; font-style: italic; margin-bottom: 18px;
    }
    .hero-card {
        background: linear-gradient(145deg, rgba(12,35,55,.95), rgba(7,23,39,.95));
        border: 1px solid #15517b; border-radius: 16px; padding: 22px 24px;
        box-shadow: 0 12px 35px rgba(0,0,0,.18);
    }
    .info-card {
        background: linear-gradient(145deg, rgba(10,31,49,.96), rgba(7,22,37,.96));
        border: 1px solid #176394; border-radius: 14px; padding: 18px 20px;
        min-height: 210px;
    }
    .workflow-box {
        background: linear-gradient(145deg, #0b2942, #081d31);
        border: 1px solid #1586cf; border-radius: 12px; padding: 16px 10px;
        text-align: center; min-height: 118px;
        display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 8px 24px rgba(0,0,0,.16);
    }
    .workflow-icon { font-size: 30px; margin-bottom: 8px; }
    .workflow-title { font-size: 14px; font-weight: 700; color: #e9f5ff; }
    .workflow-arrow { text-align: center; font-size: 27px; color: #16a7ff; padding-top: 42px; }
    .section-title { color: #f1f7ff; font-size: 22px; font-weight: 750; margin-bottom: 10px; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: #091b2c; border-radius: 12px 12px 0 0; padding: 0 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #a9c2d7; background: transparent; border-radius: 10px 10px 0 0;
        padding: 12px 20px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: white !important; background: linear-gradient(90deg, #0e87df, #0c6fc0) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0b9bf0, #0877d1);
        border: 1px solid #23aaff; border-radius: 9px;
    }
    div[data-testid="stMetric"] {
        background: #0b2034; border: 1px solid #174969; border-radius: 12px; padding: 12px;
    }
    .small-note { color: #8da8bd; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CONSTANTS
# ==============================================================================

INPUT_COLS = [
    "Power",
    "Speed",
    "ExposureTime",
    "EnergyDensity",
    "Al",
    "Fe",
    "Cr",
    "Ti",
    "Si"
]

REG_COLS = [
    "Depth",
    "Width"
]

CLS_COL = "DefectType"

REQUIRED_COLS = INPUT_COLS + REG_COLS + [CLS_COL]


# ==============================================================================
# SESSION STATE
# ==============================================================================

if "model" not in st.session_state:
    st.session_state.model = None

if "scaler" not in st.session_state:
    st.session_state.scaler = None

if "le" not in st.session_state:
    st.session_state.le = None

if "df" not in st.session_state:
    st.session_state.df = None

if "input_cols" not in st.session_state:
    st.session_state.input_cols = None

if "history" not in st.session_state:
    st.session_state.history = None

if "dataset_source" not in st.session_state:
    st.session_state.dataset_source = None


# ==============================================================================
# APPLICATION HEADER
# ==============================================================================

header_left, header_right = st.columns([8, 1])
with header_left:
    st.markdown('<div class="main-title">Optimize your LPBF Processing Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Hybrid FEM + Machine Learning Framework</div>', unsafe_allow_html=True)
with header_right:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=125)

# ==============================================================================
# SIDEBAR
# ==============================================================================

if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=190)

st.sidebar.markdown(
    '<div class="brand-card">'
    '<div class="brand-name">OPTIMUM <span>AM</span></div>'
    '<div class="brand-subtitle">AI-driven Additive Manufacturing Optimization</div>'
    '</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="section-title">⚙️ Configuration Panel</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="color:#8fc8ec;font-weight:700;margin-bottom:10px;">Model Network Hyperparameters</div>', unsafe_allow_html=True)

hidden_layers = st.sidebar.slider(
    "Number of hidden layers",
    min_value=1,
    max_value=5,
    value=2,
    step=1
)

neurons_per_layer = st.sidebar.slider(
    "Neurons per hidden layer",
    min_value=4,
    max_value=128,
    value=32,
    step=4
)

epochs = st.sidebar.slider(
    "Training epochs",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)

batch_size = st.sidebar.select_slider(
    "Batch size",
    options=[4, 8, 16, 32, 64, 128],
    value=16
)

learning_rate = st.sidebar.select_slider(
    "Learning rate",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    value=0.001,
    format_func=lambda x: f"{x:g}"
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Training Dataset")

dataset_fraction = st.sidebar.slider(
    "Dataset size used for training",
    min_value=25,
    max_value=100,
    value=100,
    step=25,
    format="%d%%"
)

st.sidebar.caption(
    "The selected percentage is sampled from the active dataset. "
    "Use 100% to train with all available rows."
)


# ==============================================================================
# TABS
# ==============================================================================

tabs = st.tabs([
    "▣  Overview",
    "▤  Data & Training",
    "⌁  Predict"
])


# ==============================================================================
# TAB 1: OVERVIEW & METHODOLOGY
# ==============================================================================

with tabs[0]:

    st.markdown('<div class="section-title">How It Works & Methodology</div>', unsafe_allow_html=True)

    intro_col, steps_col = st.columns([1.15, 0.85])

    with intro_col:
        st.markdown('''
        <div class="hero-card">
            <div style="font-size:16px;line-height:1.55;color:#d9eaf7;">
            The proposed approach combines <b>Finite Element Method (FEM)</b>
            simulations with an <b>Artificial Neural Network (ANN)</b> and
            transfer learning.
            <br><br>
            The ANN is first trained using FEM-generated data and subsequently
            fine-tuned using experimental data.
            </div>
            <div style="color:#10a7ff;font-size:18px;font-weight:750;margin-top:18px;">The Model Predicts</div>
            <ul style="color:#cfe4f4;line-height:1.8;">
                <li>Melt pool depth (μm)</li>
                <li>Melt pool width (μm)</li>
                <li>Defect type: Lack of Fusion (LoF), Full-dense, Keyhole</li>
            </ul>
        </div>
        ''', unsafe_allow_html=True)

    with steps_col:
        st.markdown('''
        <div class="info-card">
            <div style="color:#16a7ff;font-size:18px;font-weight:750;margin-bottom:12px;">ⓘ Steps to use the App</div>
            <div style="line-height:1.55;color:#d4e6f4;">
            <b>① Load Data</b><br><span class="small-note">Go to Data & Training.</span><br><br>
            <b>② Choose Dataset</b><br><span class="small-note">Use the example dataset or upload your own.</span><br><br>
            <b>③ Configure Architecture</b><br><span class="small-note">Adjust network and training settings.</span><br><br>
            <b>④ Train</b><br><span class="small-note">Train the multitask network.</span><br><br>
            <b>⑤ Predict</b><br><span class="small-note">Use the prediction tab.</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:24px;">Workflow</div>', unsafe_allow_html=True)

    workflow = [
        ("🔥", "FEM Simulations"),
        ("🧠", "Source ANN"),
        ("▤", "Pre-trained Source Model"),
        ("⚗", "Experimental Data"),
        ("🧠", "Transfer Learning / Fine-tuning"),
        ("◈", "Melt Pool & Defect Prediction"),
    ]

    cols = st.columns([1.4, .28, 1.4, .28, 1.4, .28, 1.4, .28, 1.4, .28, 1.4])
    for i, (icon, title) in enumerate(workflow):
        with cols[i * 2]:
            st.markdown(
                f'<div class="workflow-box"><div class="workflow-icon">{icon}</div>'
                f'<div class="workflow-title">{title}</div></div>',
                unsafe_allow_html=True
            )
        if i < len(workflow) - 1:
            with cols[i * 2 + 1]:
                st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2: DATA MANAGEMENT & TRAINING
# ==============================================================================

with tabs[1]:

    st.header("Data Loading & Training")


    # ==========================================================================
    # REFERENCE DATA STRUCTURE
    # ==========================================================================

    st.subheader("1. Reference Data Structure")

    st.caption(
        "Your dataset should contain the following columns:"
    )

    template_df = pd.DataFrame({
        "Power": [180, 220, 260],
        "Speed": [800, 1000, 1200],
        "ExposureTime": [40, 60, 90],
        "EnergyDensity": [55, 65, 80],
        "Al": [94.5, 90.0, 88.5],
        "Fe": [1.8, 4.0, 5.5],
        "Cr": [1.4, 2.5, 3.2],
        "Ti": [1.2, 1.8, 2.0],
        "Si": [1.1, 1.7, 0.8],
        "Depth": [120.5, 450.2, 980.1],
        "Width": [230.1, 510.4, 890.7],
        "DefectType": [
            "Full-dense",
            "LoF",
            "Keyhole"
        ]
    })

    st.dataframe(
        template_df,
        use_container_width=True,
        hide_index=True
    )


    # ==========================================================================
    # DATA SOURCE
    # ==========================================================================

    st.subheader("2. Choose Your Dataset")

    data_option = st.radio(
        "Select the source of your training data:",
        [
            "🧪 Use Example Dataset",
            "📁 Upload CSV / Excel",
            "✏️ Edit / Paste Data",
        ],
        horizontal=True
    )


    # ==========================================================================
    # EXAMPLE DATASET
    # ==========================================================================

    if data_option == "🧪 Use Example Dataset":

        example_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "example_dataset.xlsx"
        )

        if os.path.exists(example_path):

            try:

                example_df = pd.read_excel(
                    example_path,
                    engine="openpyxl"
                )

                # Remove completely empty rows
                example_df = example_df.dropna(
                    how="all"
                ).reset_index(drop=True)

                # Check columns
                missing_cols = [
                    col
                    for col in REQUIRED_COLS
                    if col not in example_df.columns
                ]

                if missing_cols:

                    st.error(
                        "The example Excel file is missing the "
                        f"following required columns:\n\n"
                        f"{missing_cols}"
                    )

                    st.info(
                        "Required columns are:\n\n"
                        + ", ".join(REQUIRED_COLS)
                    )

                else:

                    st.session_state.df = example_df
                    st.session_state.dataset_source = "Example Dataset"

                    st.success(
                        f"✅ Example dataset loaded successfully: "
                        f"{len(example_df)} rows × "
                        f"{len(example_df.columns)} columns"
                    )

            except Exception as e:

                st.error(
                    f"❌ Could not read example_dataset.xlsx\n\n"
                    f"Error: {e}"
                )

        else:

            st.error(
                "❌ example_dataset.xlsx was not found."
            )

            st.info(
                "Make sure your repository has this structure:\n\n"
                "`app.py`\n\n"
                "`example_dataset.xlsx`\n\n"
                "`requirements.txt`"
            )


    # ==========================================================================
    # UPLOAD CSV / EXCEL
    # ==========================================================================

    elif data_option == "📁 Upload CSV / Excel":

        uploaded_file = st.file_uploader(
            "Upload your experimental or FEM dataset",
            type=["csv", "xlsx"],
            help="Accepted formats: CSV and Excel (.xlsx)"
        )

        if uploaded_file is not None:

            try:

                if uploaded_file.name.lower().endswith(".xlsx"):

                    uploaded_df = pd.read_excel(
                        uploaded_file,
                        engine="openpyxl"
                    )

                else:

                    uploaded_df = pd.read_csv(
                        uploaded_file
                    )

                # Remove completely empty rows
                uploaded_df = uploaded_df.dropna(
                    how="all"
                ).reset_index(drop=True)

                # Check required columns
                missing_cols = [
                    col
                    for col in REQUIRED_COLS
                    if col not in uploaded_df.columns
                ]

                if missing_cols:

                    st.error(
                        "❌ Your dataset is missing these columns:"
                    )

                    st.write(missing_cols)

                    st.info(
                        "Required columns:"
                    )

                    st.code(
                        ", ".join(REQUIRED_COLS)
                    )

                else:

                    st.session_state.df = uploaded_df
                    st.session_state.dataset_source = (
                        uploaded_file.name
                    )

                    st.success(
                        f"✅ Dataset loaded successfully: "
                        f"{len(uploaded_df)} rows × "
                        f"{len(uploaded_df.columns)} columns"
                    )

            except Exception as e:

                st.error(
                    f"❌ Could not read the uploaded file.\n\n"
                    f"Error: {e}"
                )


    # ==========================================================================
    # MANUAL DATA
    # ==========================================================================

    elif data_option == "✏️ Edit / Paste Data":

        if st.session_state.df is None:

            st.session_state.df = template_df.copy()

            st.session_state.dataset_source = (
                "Manual Dataset"
            )

        st.info(
            "You can edit the table below or paste data "
            "directly into the cells."
        )


    # ==========================================================================
    # SYNTHETIC DATA
    # ==========================================================================

    elif data_option == "🎲 Generate Synthetic Data":

        st.sidebar.subheader("Synthetic Dataset")

        n_samples = st.sidebar.number_input(
            "Number of synthetic rows",
            min_value=10,
            max_value=5000,
            value=50,
            step=10
        )

        if st.button(
            "🎲 Generate Synthetic Dataset"
        ):

            np.random.seed(42)

            synthetic_df = pd.DataFrame({

                "Power":
                    np.random.randint(
                        100,
                        1000,
                        n_samples
                    ),

                "Speed":
                    np.random.randint(
                        100,
                        2000,
                        n_samples
                    ),

                "ExposureTime":
                    np.random.randint(
                        10,
                        200,
                        n_samples
                    ),

                "EnergyDensity":
                    np.random.uniform(
                        10,
                        200,
                        n_samples
                    ),

                "Al":
                    np.random.uniform(
                        85,
                        95,
                        n_samples
                    ),

                "Fe":
                    np.random.uniform(
                        2,
                        8,
                        n_samples
                    ),

                "Cr":
                    np.random.uniform(
                        1,
                        5,
                        n_samples
                    ),

                "Ti":
                    np.random.uniform(
                        0,
                        3,
                        n_samples
                    ),

                "Si":
                    np.random.uniform(
                        0.5,
                        3,
                        n_samples
                    ),

                "Depth":
                    np.random.uniform(
                        10,
                        2000,
                        n_samples
                    ),

                "Width":
                    np.random.uniform(
                        50,
                        3000,
                        n_samples
                    ),

                "DefectType":
                    np.random.choice(
                        [
                            "LoF",
                            "Keyhole",
                            "Full-dense"
                        ],
                        n_samples
                    )
            })

            st.session_state.df = synthetic_df

            st.session_state.dataset_source = (
                "Synthetic Dataset"
            )

            st.success(
                f"✅ Generated {n_samples} synthetic rows."
            )


    # ==========================================================================
    # ACTIVE DATASET
    # ==========================================================================

    if st.session_state.df is not None:

        st.subheader("3. Active Working Dataset")

        if st.session_state.dataset_source:

            st.caption(
                f"Current dataset: "
                f"**{st.session_state.dataset_source}**"
            )

        st.caption(
            "You can edit individual cells, paste values, "
            "or add/remove rows. The example dataset can be used directly "
            "for model training."
        )

        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            height=450,
            hide_index=True
        )

        st.session_state.df = edited_df

        # Dataset information
        st.markdown("### Dataset Information")

        info_col1, info_col2, info_col3 = st.columns(3)

        info_col1.metric(
            "Number of Samples",
            len(st.session_state.df)
        )

     

        if CLS_COL in st.session_state.df.columns:

            info_col3.metric(
                "Defect Classes",
                st.session_state.df[CLS_COL]
                .nunique()
            )


    # ==========================================================================
    # TRAINING
    # ==========================================================================

    st.subheader("4. 🚀 Train Neural Network")

    if st.session_state.df is not None:
        st.info(
            f"Training dataset: **{dataset_fraction}%** of the active dataset "
            f"({max(1, round(len(st.session_state.df) * dataset_fraction / 100))} "
            f"rows approximately). "
            "Adjust the percentage in the sidebar."
        )

    if st.session_state.df is None:

        st.warning(
            "Please load a dataset before training."
        )

    elif len(st.session_state.df) < 5:

        st.warning(
            "Please load at least 5 rows of data before training."
        )

    else:

        if st.button(
            "🚀 Train Multitask Network Model",
            type="primary"
        ):

            df = st.session_state.df.copy()

            # ------------------------------------------------------------------
            # Check columns
            # ------------------------------------------------------------------

            missing_cols = [
                col
                for col in REQUIRED_COLS
                if col not in df.columns
            ]

            if missing_cols:

                st.error(
                    "Training cannot start because these columns "
                    "are missing:"
                )

                st.write(missing_cols)

            else:

                # ------------------------------------------------------------------
                # Check missing values
                # ------------------------------------------------------------------

                training_df = df[
                    REQUIRED_COLS
                ].copy()

                # ==============================================================
                # SELECT TRAINING DATASET SIZE
                # ==============================================================

                if dataset_fraction < 100:
                    # Stratified sampling keeps the defect-class distribution
                    # approximately consistent with the active dataset.
                    sampled_parts = []

                    for class_name, class_df in training_df.groupby(
                        CLS_COL,
                        sort=False
                    ):
                        n_class = max(
                            1,
                            int(round(
                                len(class_df) * dataset_fraction / 100
                            ))
                        )

                        n_class = min(
                            n_class,
                            len(class_df)
                        )

                        sampled_parts.append(
                            class_df.sample(
                                n=n_class,
                                random_state=42
                            )
                        )

                    training_df = pd.concat(
                        sampled_parts,
                        ignore_index=True
                    ).sample(
                        frac=1,
                        random_state=42
                    ).reset_index(drop=True)

                st.write(
                    f"**Rows selected for training:** {len(training_df)} "
                    f"of {len(df)}"
                )

                missing_values = (
                    training_df.isnull()
                    .sum()
                )

                columns_with_missing = (
                    missing_values[
                        missing_values > 0
                    ]
                )

                if len(columns_with_missing) > 0:

                    st.error(
                        "Your dataset contains missing values."
                    )

                    st.write(
                        columns_with_missing
                    )

                else:

                    try:

                        with st.spinner(
                            "Processing dataset and "
                            "training neural network..."
                        ):

                            # ==================================================
                            # PREPARE INPUT DATA
                            # ==================================================

                            X = (
                                training_df[
                                    INPUT_COLS
                                ]
                                .apply(
                                    pd.to_numeric,
                                    errors="coerce"
                                )
                                .values
                            )

                            y_reg = (
                                training_df[
                                    REG_COLS
                                ]
                                .apply(
                                    pd.to_numeric,
                                    errors="coerce"
                                )
                                .values
                            )

                            y_cls_raw = (
                                training_df[
                                    CLS_COL
                                ]
                                .astype(str)
                                .values
                            )

                            # Check that the selected dataset contains enough
                            # samples from each class for classification.
                            class_counts = (
                                pd.Series(y_cls_raw)
                                .value_counts()
                            )

                            if len(class_counts) < 2:
                                raise ValueError(
                                    "At least two different DefectType classes "
                                    "are required for classification."
                                )

                            if class_counts.min() < 2:
                                raise ValueError(
                                    "Each DefectType class must contain at "
                                    "least 2 samples in the selected training "
                                    "dataset. Increase the dataset size or "
                                    "add more data."
                                )

                            # Check numeric conversion
                            if np.isnan(X).any():

                                raise ValueError(
                                    "One or more input columns "
                                    "contain non-numeric values."
                                )

                            if np.isnan(y_reg).any():

                                raise ValueError(
                                    "Depth or Width contains "
                                    "non-numeric values."
                                )

                            # ==================================================
                            # ENCODE DEFECT CLASSES
                            # ==================================================

                            le = LabelEncoder()

                            y_cls_encoded = (
                                le.fit_transform(
                                    y_cls_raw
                                )
                            )

                            num_actual_classes = (
                                len(le.classes_)
                            )

                            if num_actual_classes < 2:

                                raise ValueError(
                                    "At least two different "
                                    "DefectType classes are required "
                                    "for classification."
                                )

                            y_cls_onehot = (
                                to_categorical(
                                    y_cls_encoded,
                                    num_classes=
                                    num_actual_classes
                                )
                            )

                            # ==================================================
                            # NORMALIZATION
                            # ==================================================

                            scaler = StandardScaler()

                            X_scaled = (
                                scaler.fit_transform(X)
                            )

                            # ==================================================
                            # TRAIN / TEST SPLIT
                            # ==================================================

                            (
                                X_train,
                                X_test,
                                y_reg_train,
                                y_reg_test,
                                y_cls_train,
                                y_cls_test
                            ) = train_test_split(
                                X_scaled,
                                y_reg,
                                y_cls_onehot,
                                test_size=0.2,
                                random_state=42
                            )

                            # ==================================================
                            # BUILD NETWORK
                            # ==================================================

                            input_layer = Input(
                                shape=(len(INPUT_COLS),),
                                name="process_material_input"
                            )

                            shared = input_layer

                            for i in range(
                                hidden_layers
                            ):

                                shared = Dense(
                                    neurons_per_layer,
                                    activation="relu",
                                    name=
                                    f"shared_dense_{i+1}"
                                )(shared)

                            # ==================================================
                            # REGRESSION HEAD
                            # ==================================================

                            reg_branch = Dense(
                                max(
                                    neurons_per_layer // 2,
                                    4
                                ),
                                activation="relu",
                                name="reg_dense"
                            )(shared)

                            reg_output = Dense(
                                len(REG_COLS),
                                activation="linear",
                                name="reg_output"
                            )(reg_branch)

                            # ==================================================
                            # CLASSIFICATION HEAD
                            # ==================================================

                            cls_branch = Dense(
                                max(
                                    neurons_per_layer // 2,
                                    4
                                ),
                                activation="relu",
                                name="cls_dense"
                            )(shared)

                            cls_output = Dense(
                                num_actual_classes,
                                activation="softmax",
                                name="cls_output"
                            )(cls_branch)

                            # ==================================================
                            # CREATE MODEL
                            # ==================================================

                            model = Model(
                                inputs=input_layer,
                                outputs=[
                                    reg_output,
                                    cls_output
                                ]
                            )

                            optimizer = tf.keras.optimizers.Adam(
                                learning_rate=learning_rate
                            )

                            model.compile(
                                optimizer=optimizer,

                                loss={
                                    "reg_output": "mse",
                                    "cls_output":
                                        "categorical_crossentropy"
                                },

                                loss_weights={
                                    "reg_output": 1.0,
                                    "cls_output": 1.0
                                },

                                metrics={
                                    "reg_output": "mae",
                                    "cls_output": "accuracy"
                                }
                            )

                            # ==================================================
                            # EARLY STOPPING
                            # ==================================================

                            early_stop = EarlyStopping(
                                monitor="val_loss",
                                patience=20,
                                restore_best_weights=True
                            )

                            # ==================================================
                            # TRAIN
                            # ==================================================

                            # The batch size is selected from the sidebar.
                            # For very small datasets, keep it within the
                            # available number of training samples.
                            effective_batch_size = min(
                                batch_size,
                                len(X_train)
                            )

                            history = model.fit(
                                X_train,

                                {
                                    "reg_output":
                                        y_reg_train,
                                    "cls_output":
                                        y_cls_train
                                },

                                validation_data=(
                                    X_test,

                                    {
                                        "reg_output":
                                            y_reg_test,
                                        "cls_output":
                                            y_cls_test
                                    }
                                ),

                                epochs=epochs,

                                batch_size=effective_batch_size,

                                callbacks=[
                                    early_stop
                                ],

                                verbose=0
                            )

                            # ==================================================
                            # SAVE MODEL ASSETS
                            # ==================================================

                            st.session_state.model = model

                            st.session_state.scaler = scaler

                            st.session_state.le = le

                            st.session_state.input_cols = (
                                INPUT_COLS
                            )

                            st.session_state.history = (
                                history.history
                            )

                        # ======================================================
                        # TRAINING SUCCESS
                        # ======================================================

                        st.success(
                            f"✅ Training completed successfully! "
                            f"Training stopped after "
                            f"{len(history.history['loss'])} epochs."
                        )

                        # ======================================================
                        # TRAINING CURVES
                        # ======================================================

                        st.subheader(
                            "📈 Training Performance"
                        )

                        fig, axes = plt.subplots(
                            1,
                            3,
                            figsize=(16, 4)
                        )

                        axes[0].plot(
                            history.history["loss"],
                            label="Train"
                        )

                        axes[0].plot(
                            history.history["val_loss"],
                            label="Validation"
                        )

                        axes[0].set_title(
                            "Total Loss"
                        )

                        axes[0].set_xlabel(
                            "Epoch"
                        )

                        axes[0].set_ylabel(
                            "Loss"
                        )

                        axes[0].legend()

                        axes[1].plot(
                            history.history[
                                "reg_output_mae"
                            ],
                            label="Train"
                        )

                        axes[1].plot(
                            history.history[
                                "val_reg_output_mae"
                            ],
                            label="Validation"
                        )

                        axes[1].set_title(
                            "Regression MAE"
                        )

                        axes[1].set_xlabel(
                            "Epoch"
                        )

                        axes[1].set_ylabel(
                            "MAE"
                        )

                        axes[1].legend()

                        axes[2].plot(
                            history.history[
                                "cls_output_accuracy"
                            ],
                            label="Train"
                        )

                        axes[2].plot(
                            history.history[
                                "val_cls_output_accuracy"
                            ],
                            label="Validation"
                        )

                        axes[2].set_title(
                            "Defect Classification Accuracy"
                        )

                        axes[2].set_xlabel(
                            "Epoch"
                        )

                        axes[2].set_ylabel(
                            "Accuracy"
                        )

                        axes[2].legend()

                        st.pyplot(
                            fig,
                            use_container_width=True
                        )

                        plt.close(fig)

                        # ======================================================
                        # TEST METRICS
                        # ======================================================

                        st.subheader(
                            "📊 Held-out Test Metrics"
                        )

                        test_results = model.evaluate(
                            X_test,

                            {
                                "reg_output":
                                    y_reg_test,
                                "cls_output":
                                    y_cls_test
                            },

                            verbose=0
                        )

                        metric_names = (
                            model.metrics_names
                        )

                        metrics_dict = {
                            name: round(
                                float(value),
                                4
                            )

                            for name, value
                            in zip(
                                metric_names,
                                test_results
                            )
                        }

                        st.write(
                            metrics_dict
                        )

                        # ======================================================
                        # MODEL SUMMARY
                        # ======================================================

                        with st.expander(
                            "🔧 Show Model Architecture"
                        ):

                            model_summary = []

                            model.summary(
                                print_fn=lambda x:
                                    model_summary.append(x)
                            )

                            st.code(
                                "\n".join(
                                    model_summary
                                )
                            )

                        st.info(
                            "The trained model is now available "
                            "in the ⌁ Prediction tab."
                        )

                    except Exception as e:

                        st.error(
                            f"❌ Training failed:\n\n{e}"
                        )


# ==============================================================================
# TAB 3: INFERENCE / PREDICTION
# ==============================================================================

with tabs[2]:

    st.header(
        "🔮 Predict Melt Pool Geometry & Defect Type"
    )

    if st.session_state.model is None:

        st.warning(
            "No trained model found yet. "
            "Go to the ▤ Data & Training tab "
            "and train a model first."
        )

    else:

        st.success(
            "✅ Trained model is ready for prediction."
        )

        st.caption(
            "Enter a new laser process and alloy "
            "composition to predict melt pool depth, "
            "width, and defect type."
        )

        # ======================================================================
        # INPUTS
        # ======================================================================

        col1, col2, col3 = st.columns(3)

        with col1:

            power = st.number_input(
                "Power (W)",
                min_value=0.0,
                value=200.0,
                step=5.0
            )

            speed = st.number_input(
                "Scan Speed (mm/s)",
                min_value=0.0,
                value=1000.0,
                step=10.0
            )

            exposure = st.number_input(
                "Exposure Time (μs)",
                min_value=0.0,
                value=60.0,
                step=1.0
            )

        with col2:

            energy_density = st.number_input(
                "Energy Density (J/mm³)",
                min_value=0.0,
                value=60.0,
                step=1.0
            )

            al = st.number_input(
                "Al (wt.%)",
                min_value=0.0,
                max_value=100.0,
                value=90.0,
                step=0.1
            )

            fe = st.number_input(
                "Fe (wt.%)",
                min_value=0.0,
                max_value=100.0,
                value=4.0,
                step=0.1
            )

        with col3:

            cr = st.number_input(
                "Cr (wt.%)",
                min_value=0.0,
                max_value=100.0,
                value=2.5,
                step=0.1
            )

            ti = st.number_input(
                "Ti (wt.%)",
                min_value=0.0,
                max_value=100.0,
                value=1.5,
                step=0.1
            )

            si = st.number_input(
                "Si (wt.%)",
                min_value=0.0,
                max_value=100.0,
                value=1.5,
                step=0.1
            )


        # ======================================================================
        # PREDICT
        # ======================================================================

        if st.button(
            "🔮 Predict",
            type="primary"
        ):

            try:

                raw_input = pd.DataFrame(
                    [[
                        power,
                        speed,
                        exposure,
                        energy_density,
                        al,
                        fe,
                        cr,
                        ti,
                        si
                    ]],

                    columns=st.session_state.input_cols
                )

                X_new = (
                    st.session_state.scaler
                    .transform(
                        raw_input.values
                    )
                )

                reg_pred, cls_pred = (
                    st.session_state.model.predict(
                        X_new,
                        verbose=0
                    )
                )

                # ==============================================================
                # REGRESSION
                # ==============================================================

                depth_pred = reg_pred[0][0]

                width_pred = reg_pred[0][1]

                # ==============================================================
                # CLASSIFICATION
                # ==============================================================

                defect_idx = int(
                    np.argmax(
                        cls_pred[0]
                    )
                )

                defect_label = (
                    st.session_state.le
                    .inverse_transform(
                        [defect_idx]
                    )[0]
                )

                confidence = (
                    float(
                        cls_pred[0][defect_idx]
                    ) * 100
                )

                # ==============================================================
                # RESULTS
                # ==============================================================

                st.subheader(
                    "🎯 Prediction Results"
                )

                res_col1, res_col2, res_col3 = (
                    st.columns(3)
                )

                res_col1.metric(
                    "Predicted Depth (μm)",
                    f"{depth_pred:.1f}"
                )

                res_col2.metric(
                    "Predicted Width (μm)",
                    f"{width_pred:.1f}"
                )

                res_col3.metric(
                    "Predicted Defect Type",
                    defect_label,
                    f"{confidence:.1f}% confidence"
                )

                # ==============================================================
                # CLASS PROBABILITIES
                # ==============================================================

                st.subheader(
                    "📊 Defect Class Probabilities"
                )

                prob_df = pd.DataFrame({

                    "DefectType":
                        st.session_state.le.classes_,

                    "Probability":
                        cls_pred[0]

                }).sort_values(
                    "Probability",
                    ascending=False
                )

                prob_df["Probability (%)"] = (
                    prob_df["Probability"] * 100
                )

                st.dataframe(
                    prob_df[
                        [
                            "DefectType",
                            "Probability (%)"
                        ]
                    ].style.format(
                        {
                            "Probability (%)":
                                "{:.2f}%"
                        }
                    ),

                    use_container_width=True,

                    hide_index=True
                )

                st.bar_chart(
                    prob_df.set_index(
                        "DefectType"
                    )["Probability"]
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")

st.markdown(
    "<div style='text-align:center; color:#777; font-size:12px;'>"
    "OPTIMUM AM · AI-driven Additive Manufacturing Optimization"
    "</div>",
    unsafe_allow_html=True
)
