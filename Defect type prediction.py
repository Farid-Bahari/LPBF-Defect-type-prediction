import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LPBF Melt Pool & Defect Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    .info-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        background-color: #f5f7fa;
        border: 1px solid #e1e5eb;
        margin-bottom: 1rem;
    }

    .step-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #fafafa;
        border: 1px solid #e5e5e5;
        min-height: 120px;
    }

    .result-box {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: #f7f9fc;
        border: 1px solid #dfe5ec;
        text-align: center;
    }

    .result-number {
        font-size: 1.6rem;
        font-weight: 700;
    }

    .result-label {
        font-size: 0.9rem;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

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

EXPECTED_COLUMNS = INPUT_COLS + REG_COLS + [CLS_COL]


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "model" not in st.session_state:
    st.session_state.model = None

if "scaler" not in st.session_state:
    st.session_state.scaler = None

if "le" not in st.session_state:
    st.session_state.le = None

if "history" not in st.session_state:
    st.session_state.history = None

if "trained" not in st.session_state:
    st.session_state.trained = False


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔬 LPBF Melt Pool & Defect Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Hybrid FEM + Machine Learning framework for predicting melt pool dimensions '
    'and LPBF defect type.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HOW IT WORKS
# ============================================================

with st.expander("ℹ️ How does this application work?", expanded=True):

    st.markdown(
        """
        ### Overview

        This application uses a machine-learning model developed from LPBF
        process and material information to predict:

        - **Melt pool depth**
        - **Melt pool width**
        - **Defect type**
          - Lack of Fusion (LoF)
          - Full-dense
          - Keyhole

        ### How to use the application

        **Step 1 — Prepare your data**

        Your dataset should contain the following columns:

        **Input parameters**
        - Power
        - Speed
        - ExposureTime
        - EnergyDensity
        - Al
        - Fe
        - Cr
        - Ti
        - Si

        **Experimental outputs**
        - Depth
        - Width
        - DefectType

        **Step 2 — Load your data**

        You can:

        1. Upload an Excel (`.xlsx`) or CSV (`.csv`) file.
        2. Paste CSV data directly into the application.
        3. Enter or modify the data manually in the editable table.

        **Step 3 — Train the model**

        The application automatically preprocesses your data, splits it into
        training and validation/test sets, and trains a multitask neural network.

        The model simultaneously learns:

        - Regression → melt pool depth and width
        - Classification → defect type

        **Step 4 — Predict**

        After training, enter a new LPBF processing condition and alloy composition.
        The trained model will predict the melt pool dimensions and defect regime.

        > **Important:** Your dataset should contain experimental observations with
        > measured melt pool dimensions and defect classifications. The model
        > learns from the data you provide.
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Application")

st.sidebar.markdown(
    """
    **Workflow**

    1. Load your dataset
    2. Check the data
    3. Configure the model
    4. Train
    5. Predict
    """
)


# ============================================================
# STEP 1 — DATA INPUT
# ============================================================

st.header("1️⃣ Load Your Dataset")

st.write(
    "Upload your experimental dataset, paste CSV data, or enter your data manually."
)

data_method = st.radio(
    "Choose how you want to provide your data:",
    [
        "Upload Excel / CSV",
        "Paste CSV data",
        "Enter data manually"
    ],
    horizontal=True
)


# ------------------------------------------------------------
# OPTION 1 — UPLOAD FILE
# ------------------------------------------------------------

if data_method == "Upload Excel / CSV":

    uploaded_file = st.file_uploader(
        "Upload your dataset",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel file containing your LPBF experimental data."
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(".csv"):
                uploaded_df = pd.read_csv(uploaded_file)

            else:
                uploaded_df = pd.read_excel(uploaded_file)

            st.session_state.df = uploaded_df.copy()

            st.success(
                f"Dataset loaded successfully: "
                f"{uploaded_df.shape[0]} rows × {uploaded_df.shape[1]} columns."
            )

        except Exception as e:

            st.error(
                f"Could not read the uploaded file.\n\nError: {e}"
            )


# ------------------------------------------------------------
# OPTION 2 — PASTE CSV
# ------------------------------------------------------------

elif data_method == "Paste CSV data":

    st.info(
        "Paste your CSV data below. The first row must contain the column names."
    )

    pasted_data = st.text_area(
        "Paste CSV data here",
        height=250,
        placeholder=(
            "Power,Speed,ExposureTime,EnergyDensity,Al,Fe,Cr,Ti,Si,Depth,Width,DefectType\n"
            "200,800,85,73,94.5,1.8,1.4,1.2,1.1,850,1200,Full-dense\n"
            "180,900,80,65,94.5,1.8,1.4,1.2,1.1,500,1000,LoF"
        )
    )

    if st.button("Load pasted data"):

        if pasted_data.strip():

            try:

                pasted_df = pd.read_csv(
                    pd.io.common.StringIO(pasted_data)
                )

                st.session_state.df = pasted_df.copy()

                st.success(
                    f"Data loaded successfully: "
                    f"{pasted_df.shape[0]} rows × {pasted_df.shape[1]} columns."
                )

            except Exception as e:

                st.error(
                    f"Could not read the pasted data.\n\nError: {e}"
                )

        else:

            st.warning("Please paste your data first.")


# ------------------------------------------------------------
# OPTION 3 — MANUAL DATA
# ------------------------------------------------------------

elif data_method == "Enter data manually":

    st.info(
        "Create or edit your dataset directly in the table below."
    )

    if st.session_state.df is None:

        st.session_state.df = pd.DataFrame(
            columns=EXPECTED_COLUMNS
        )

    manual_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True
    )

    st.session_state.df = manual_df


# ============================================================
# DATA VALIDATION AND EDITOR
# ============================================================

if st.session_state.df is not None:

    st.divider()

    st.subheader("📊 Dataset")

    df = st.session_state.df.copy()

    # Show dataset
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="main_data_editor"
    )

    st.session_state.df = edited_df

    df = st.session_state.df

    # --------------------------------------------------------
    # DATA SUMMARY
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", len(df))

    with col2:
        st.metric("Columns", len(df.columns))

    with col3:
        st.metric(
            "Missing values",
            int(df.isna().sum().sum())
        )

    with col4:
        if CLS_COL in df.columns:
            st.metric(
                "Defect classes",
                df[CLS_COL].nunique()
            )
        else:
            st.metric("Defect classes", "—")


    # ========================================================
    # COLUMN VALIDATION
    # ========================================================

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:

        st.error(
            "The following required columns are missing:\n\n"
            + ", ".join(missing_columns)
        )

        st.info(
            "Please rename your columns to match the required names exactly."
        )

        st.stop()


    # ========================================================
    # DATA QUALITY CHECK
    # ========================================================

    st.subheader("🔎 Data Quality Check")

    numeric_columns = INPUT_COLS + REG_COLS

    numeric_conversion_failed = []

    for col in numeric_columns:

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if converted.isna().sum() > df[col].isna().sum():
            numeric_conversion_failed.append(col)

    if numeric_conversion_failed:

        st.error(
            "Some numeric columns contain non-numeric values: "
            + ", ".join(numeric_conversion_failed)
        )

    missing_values = df[EXPECTED_COLUMNS].isna().sum()

    if missing_values.sum() > 0:

        st.warning("Missing values were found in the required columns:")

        st.dataframe(
            missing_values[missing_values > 0],
            use_container_width=True
        )

    else:

        st.success("✓ No missing values detected in the required columns.")


    # ========================================================
    # DEFECT CLASS CHECK
    # ========================================================

    defect_values = (
        df[CLS_COL]
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    st.write(
        "**Defect classes detected:**",
        ", ".join(defect_values)
    )

    expected_defects = {
        "LoF",
        "Keyhole",
        "Full-dense"
    }

    missing_defects = expected_defects - set(defect_values)

    if missing_defects:

        st.warning(
            "The dataset does not contain all three expected defect classes: "
            + ", ".join(sorted(missing_defects))
        )


    # ========================================================
    # MODEL CONFIGURATION
    # ========================================================

    st.divider()

    st.header("2️⃣ Configure Model Training")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        hidden_layers = st.number_input(
            "Hidden layers",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )

    with col2:

        neurons_per_layer = st.number_input(
            "Neurons per layer",
            min_value=4,
            max_value=256,
            value=32,
            step=4
        )

    with col3:

        epochs = st.number_input(
            "Maximum epochs",
            min_value=10,
            max_value=1000,
            value=150,
            step=10
        )

    with col4:

        test_size = st.slider(
            "Validation/test fraction",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05
        )


    st.info(
        "The network is a multitask model: it predicts melt pool depth and width "
        "as regression outputs and defect type as a classification output."
    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    st.header("3️⃣ Train Model")

    train_button = st.button(
        "🚀 Train Model",
        type="primary",
        use_container_width=True
    )

    if train_button:

        # ----------------------------------------------------
        # Check data
        # ----------------------------------------------------

        training_df = df[EXPECTED_COLUMNS].copy()

        # Convert numerical columns
        for col in numeric_columns:

            training_df[col] = pd.to_numeric(
                training_df[col],
                errors="coerce"
            )

        # Remove incomplete rows
        training_df = training_df.dropna(
            subset=EXPECTED_COLUMNS
        )

        if len(training_df) < 20:

            st.error(
                "At least 20 complete observations are recommended for training."
            )

            st.stop()


        # ----------------------------------------------------
        # Check classes
        # ----------------------------------------------------

        class_counts = (
            training_df[CLS_COL]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        if len(class_counts) < 2:

            st.error(
                "At least two different defect classes are required for classification."
            )

            st.stop()

        if class_counts.min() < 2:

            st.error(
                "Each defect class should contain at least two observations."
            )

            st.stop()


        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        X = training_df[INPUT_COLS].values.astype("float64")

        y_reg = (
            training_df[REG_COLS]
            .values
            .astype("float64")
        )

        y_cls_raw = (
            training_df[CLS_COL]
            .astype(str)
            .str.strip()
            .values
        )


        # ----------------------------------------------------
        # Encode labels
        # ----------------------------------------------------

        le = LabelEncoder()

        y_cls_encoded = le.fit_transform(
            y_cls_raw
        )

        number_of_classes = len(
            le.classes_
        )

        y_cls_onehot = to_categorical(
            y_cls_encoded,
            num_classes=number_of_classes
        )


        # ----------------------------------------------------
        # Scale input
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(
            X
        )


        # ----------------------------------------------------
        # Train/test split
        # ----------------------------------------------------

        try:

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
                test_size=test_size,
                random_state=42,
                stratify=y_cls_encoded
            )

        except ValueError:

            st.error(
                "The dataset is too small or imbalanced for the selected "
                "validation/test fraction. Please add more data or adjust "
                "the test fraction."
            )

            st.stop()


        # ====================================================
        # BUILD MODEL
        # ====================================================

        seed = 42

        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


        input_layer = Input(
            shape=(len(INPUT_COLS),),
            name="input"
        )

        shared = input_layer

        for i in range(hidden_layers):

            shared = Dense(
                int(neurons_per_layer),
                activation="relu",
                name=f"hidden_{i+1}"
            )(shared)


        # Regression output
        reg_output = Dense(
            2,
            activation="relu",
            name="regression"
        )(shared)


        # Classification output
        cls_output = Dense(
            number_of_classes,
            activation="softmax",
            name="classification"
        )(shared)


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
                "regression": "mse",
                "classification": "categorical_crossentropy"
            },
            metrics={
                "regression": ["mae"],
                "classification": ["accuracy"]
            }
        )


        # ====================================================
        # TRAINING
        # ====================================================

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        )


        progress_bar = st.progress(0)

        status_text = st.empty()

        status_text.info(
            "Training model..."
        )


        history = model.fit(
            X_train,
            {
                "regression": y_reg_train,
                "classification": y_cls_train
            },
            validation_data=(
                X_test,
                {
                    "regression": y_reg_test,
                    "classification": y_cls_test
                }
            ),
            epochs=int(epochs),
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )


        progress_bar.progress(100)

        status_text.success(
            "✓ Model training completed."
        )


        # ----------------------------------------------------
        # Save in session
        # ----------------------------------------------------

        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.le = le
        st.session_state.history = history
        st.session_state.trained = True


        # ====================================================
        # TRAINING RESULTS
        # ====================================================

        st.success(
            f"Model trained using {len(training_df)} observations."
        )


        st.subheader("📈 Training Performance")


        history_dict = history.history


        fig1 = plt.figure(figsize=(8, 5))

        plt.plot(
            history_dict["regression_mae"],
            label="Training MAE"
        )

        plt.plot(
            history_dict["val_regression_mae"],
            label="Validation MAE"
        )

        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.title("Melt Pool Regression Performance")
        plt.legend()
        plt.tight_layout()

        st.pyplot(
            fig1
        )

        plt.close(fig1)


        fig2 = plt.figure(figsize=(8, 5))

        plt.plot(
            history_dict["classification_accuracy"],
            label="Training Accuracy"
        )

        plt.plot(
            history_dict["val_classification_accuracy"],
            label="Validation Accuracy"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Defect Classification Performance")
        plt.legend()
        plt.tight_layout()

        st.pyplot(
            fig2
        )

        plt.close(fig2)


        # ----------------------------------------------------
        # Model information
        # ----------------------------------------------------

        st.subheader("Model Information")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Training samples",
                len(X_train)
            )

        with c2:
            st.metric(
                "Validation/Test samples",
                len(X_test)
            )

        with c3:
            st.metric(
                "Defect classes",
                len(le.classes_)
            )

        st.write(
            "**Detected classes:**",
            ", ".join(le.classes_)
        )


# ============================================================
# PREDICTION
# ============================================================

if st.session_state.trained:

    st.divider()

    st.header("4️⃣ Predict a New LPBF Condition")

    st.write(
        "Enter the material composition and LPBF processing parameters "
        "to obtain a prediction from your trained model."
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Keep exactly the same order as INPUT_COLS
    # --------------------------------------------------------

    st.subheader("Material Composition")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        al = st.number_input(
            "Al [wt.%]",
            min_value=0.0,
            max_value=100.0,
            value=94.5,
            step=0.1
        )

    with col2:
        fe = st.number_input(
            "Fe [wt.%]",
            min_value=0.0,
            max_value=100.0,
            value=1.8,
            step=0.1
        )

    with col3:
        cr = st.number_input(
            "Cr [wt.%]",
            min_value=0.0,
            max_value=100.0,
            value=1.4,
            step=0.1
        )

    with col4:
        ti = st.number_input(
            "Ti [wt.%]",
            min_value=0.0,
            max_value=100.0,
            value=1.2,
            step=0.1
        )

    with col5:
        si = st.number_input(
            "Si [wt.%]",
            min_value=0.0,
            max_value=100.0,
            value=1.1,
            step=0.1
        )


    st.subheader("LPBF Processing Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        power = st.number_input(
            "Laser Power [W]",
            min_value=0.0,
            max_value=5000.0,
            value=200.0,
            step=10.0
        )

    with col2:
        speed = st.number_input(
            "Scan Speed [mm/s]",
            min_value=0.0,
            max_value=10000.0,
            value=842.0,
            step=10.0
        )

    with col3:
        et = st.number_input(
            "Exposure Time [µs]",
            min_value=0.0,
            max_value=10000.0,
            value=85.0,
            step=5.0
        )

    with col4:
        ed = st.number_input(
            "Energy Density [J/mm³]",
            min_value=0.0,
            max_value=10000.0,
            value=73.0,
            step=1.0
        )


    st.write("")

    predict_button = st.button(
        "🔮 Predict",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        # ----------------------------------------------------
        # EXACT SAME FEATURE ORDER AS TRAINING
        # ----------------------------------------------------

        X_new = pd.DataFrame(
            [[
                power,
                speed,
                et,
                ed,
                al,
                fe,
                cr,
                ti,
                si
            ]],
            columns=INPUT_COLS
        )


        X_new_scaled = (
            st.session_state.scaler
            .transform(X_new)
        )


        y_reg_pred, y_cls_pred = (
            st.session_state.model
            .predict(
                X_new_scaled,
                verbose=0
            )
        )


        depth_prediction = float(
            y_reg_pred[0, 0]
        )

        width_prediction = float(
            y_reg_pred[0, 1]
        )


        predicted_class_index = int(
            np.argmax(
                y_cls_pred[0]
            )
        )


        defect_prediction = (
            st.session_state.le
            .inverse_transform(
                [predicted_class_index]
            )[0]
        )


        confidence = float(
            np.max(
                y_cls_pred[0]
            )
        )


        # ====================================================
        # RESULTS
        # ====================================================

        st.subheader("🎯 Prediction Results")


        col1, col2, col3 = st.columns(3)


        with col1:

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">
                        Melt Pool Depth
                    </div>
                    <div class="result-number">
                        {depth_prediction:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">
                        Melt Pool Width
                    </div>
                    <div class="result-number">
                        {width_prediction:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                f"""
                <div class="result-box">
                    <div class="result-label">
                        Predicted Defect Type
                    </div>
                    <div class="result-number">
                        {defect_prediction}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


        st.write("")

        st.write(
            f"**Classification confidence:** {confidence:.1%}"
        )


        # ----------------------------------------------------
        # CLASS PROBABILITIES
        # ----------------------------------------------------

        st.subheader("Defect Class Probabilities")

        probability_df = pd.DataFrame(
            {
                "Defect Type": st.session_state.le.classes_,
                "Probability": y_cls_pred[0]
            }
        )

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LPBF Melt Pool & Defect Prediction | "
    "Hybrid FEM + Machine Learning Research Application"
)
