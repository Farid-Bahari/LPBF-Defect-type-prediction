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

st.set_page_config(
    page_title="Optimize your LPBF Processing Parameters",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


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

st.title("🔥 Optimize your LPBF Processing Parameters")

st.markdown(
    "### *Hybrid FEM + Machine Learning Framework*"
)

st.write("---")


# ==============================================================================
# SIDEBAR
# ==============================================================================

st.sidebar.header("⚙️ Configuration Panel")

st.sidebar.subheader("Model Network Hyperparameters")

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
    "Training Epochs",
    min_value=10,
    max_value=500,
    value=100,
    step=10
)


# ==============================================================================
# TABS
# ==============================================================================

tabs = st.tabs([
    "📋 Overview & Methodology",
    "📊 Import data and train the model",
    "🔮 Predict"
])


# ==============================================================================
# TAB 1: OVERVIEW & METHODOLOGY
# ==============================================================================

with tabs[0]:

    st.header("How It Works & Methodology")

    col1, col2 = st.columns([3, 2])

    with col1:

        st.markdown("""
        The proposed approach combines **Finite Element Method (FEM)**
        simulations with an **Artificial Neural Network (ANN)** and
        transfer learning.

        The ANN is first trained using FEM-generated data and subsequently
        fine-tuned using experimental data.

        #### **The Model Predicts:**

        * **Melt pool depth** (μm)
        * **Melt pool width** (μm)
        * **Defect type:**
            * *Lack of Fusion (LoF)*
            * *Full-dense*
            * *Keyhole*
        """)

    with col2:

        st.info("""
        #### **Steps to use the App**

        1. **Load Data**
           Go to the *Data Management & Training* tab.

        2. **Choose Dataset**
           Use the example dataset or upload your own data.

        3. **Configure Architecture**
           Adjust neurons, hidden layers, and training epochs.

        4. **Train**
           Click *Train Multitask Network Model*.

        5. **Predict**
           Use the *Inference / Prediction* tab.
        """)

    st.subheader("Workflow Diagram")

    st.markdown("""
    ```text
    FEM Simulations
           ↓
    Source ANN
           ↓
    Pre-trained Source Model
           ↓
    Experimental Data
           ↓
    Transfer Learning / Fine-tuning
           ↓
    Melt Pool & Defect Prediction
    ```
    """)


# ==============================================================================
# TAB 2: DATA MANAGEMENT & TRAINING
# ==============================================================================

with tabs[1]:

    st.header("📊 Data Loading & training the target model")


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
            "🎲 Generate Synthetic Data"
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
            "or add/remove rows."
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

                            model.compile(
                                optimizer="adam",

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

                            batch_size = min(
                                16,
                                max(
                                    2,
                                    len(X_train) // 4
                                )
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

                                batch_size=batch_size,

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
                            "in the 🔮 Inference / Prediction tab."
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
            "Go to the 📊 Data Management & Training tab "
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
