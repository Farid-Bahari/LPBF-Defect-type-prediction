import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Optimize your LPBF processing parameters",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session States safely
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

# --- Application Header ---
st.title("🔥 Optimize your LPBF processing parameters")
st.markdown("### *Hybrid FEM + Machine Learning Framework*")
st.write("---")

# --- Tabs for Clean Navigation ---
tabs = st.tabs(["📋 Overview & Methodology", "📊 Data Management & Training", "🔮 Inference / Prediction"])

# ==============================================================================
# TAB 1: OVERVIEW & METHODOLOGY
# ==============================================================================
with tabs[0]:
    st.header("How It Works & Methodology")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        The proposed approach combines **Finite Element Method (FEM)** simulations with an **Artificial Neural Network (ANN)** and transfer learning.
        The ANN is first trained using FEM-generated data and subsequently fine-tuned using experimental data.

        #### **The Model Predicts:**
        *   **Melt pool depth** (μm)
        *   **Melt pool width** (μm)
        *   **Defect type:**
            *   *Lack of Fusion (LoF)*
            *   *Full-dense*
            *   *Keyhole*
        """)

    with col2:
        st.info("""
        #### **Steps to use the App:**
        1. **Load Data:** Go to the *Data Management & Training* tab to upload, paste or generate data.
        2. **Configure Architecture:** Adjust neurons, hidden layers, and training epochs in the sidebar.
        3. **Train:** Click *Train Model* and monitor the loss metrics.
        4. **Predict:** Open the *Inference / Prediction* tab to test new laser and material combinations.
        """)

    st.subheader("Workflow Diagram")
    st.markdown("""
    ```mermaid
    graph LR
        A[FEM Simulations] --> B(Source ANN)
        B --> C[Pre-trained Source Model]
        C --> D(Experimental Data)
        D --> E[Transfer Learning / Fine-tuning]
        E --> F[Melt Pool & Defect Prediction]

        style A fill:#f9f,stroke:#333,stroke-width:2px
        style F fill:#bbf,stroke:#333,stroke-width:2px
    ```
    """)

# ==============================================================================
# TAB 2: DATA MANAGEMENT & TRAINING
# ==============================================================================
with tabs[1]:
    st.header("Data Loading & Model Optimization")

    # 1. Base Example Structure Presentation
    st.subheader("1. Reference Data Structure")
    st.caption("Your input dataset must precisely match this sequence of features and targets:")

    # Example rows for the target schema template
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
        "DefectType": ["Full-dense", "LoF", "Keyhole"]
    })
    st.dataframe(template_df, use_container_width=True)

    # 2. Flexible Import Options
    st.subheader("2. Source Your Dataset")
    data_option = st.radio(
        "Choose how you want to provide your training dataset:",
        ["Use Synthetic Data Generator", "Upload a CSV Data File", "Manually Paste / Edit Table Below"],
        horizontal=True
    )

    # Sidebar parameter configuration
    st.sidebar.header("⚙️ Configuration Panel")
    st.sidebar.subheader("Dataset Settings")
    n_samples = st.sidebar.number_input("Number of synthetic rows to generate", min_value=10, max_value=500, value=50, step=10)

    # Handling selected option logic
    if data_option == "Use Synthetic Data Generator":
        if st.button("Generate Synthetic Data Blueprint") or st.session_state.df is None:
            np.random.seed(42)
            st.session_state.df = pd.DataFrame({
                "Power": np.random.randint(100, 1000, n_samples),
                "Speed": np.random.randint(100, 2000, n_samples),
                "ExposureTime": np.random.randint(10, 200, n_samples),
                "EnergyDensity": np.random.randint(10, 200, n_samples),
                "Al": np.random.uniform(85, 95, n_samples),
                "Fe": np.random.uniform(2, 8, n_samples),
                "Cr": np.random.uniform(1, 5, n_samples),
                "Ti": np.random.uniform(0, 3, n_samples),
                "Si": np.random.uniform(0.5, 3, n_samples),
                "Depth": np.random.uniform(10, 2000, n_samples),
                "Width": np.random.uniform(50, 3000, n_samples),
                "DefectType": np.random.choice(["LoF", "Keyhole", "Full-dense"], n_samples)
            })
            st.success("Generated dummy data environment successfully!")

    elif data_option == "Upload a CSV Data File":
        uploaded_file = st.file_uploader("Upload Your Experimental/FEM Data File (CSV)", type=["csv"])
        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file)
            required_cols = list(template_df.columns)
            if all(col in uploaded_df.columns for col in required_cols):
                st.session_state.df = uploaded_df
                st.success("File uploaded successfully and verified structure!")
            else:
                st.error(f"Missing columns! Make sure your file contains exactly: {required_cols}")

    elif data_option == "Manually Paste / Edit Table Below":
        if st.session_state.df is None:
            st.session_state.df = template_df.copy()

    # Interactive Table Editor for all modes
    st.subheader("3. Active Working Dataset Grid")
    st.caption("You can click individual cells to overwrite parameters, paste tabular text blocks directly, or add rows using the table options below:")

    edited_df = st.data_editor(
        st.session_state.df if st.session_state.df is not None else template_df,
        num_rows="dynamic",
        use_container_width=True
    )
    st.session_state.df = edited_df

    # 3. Model Architecture Parameters
    st.sidebar.subheader("Model Network Hyperparameters")
    hidden_layers = st.sidebar.slider("Number of hidden layers", 1, 5, 2, step=1)
    neurons_per_layer = st.sidebar.slider("Neurons per hidden layer", 4, 128, 32, step=4)
    epochs = st.sidebar.slider("Training Epochs", 10, 500, 100, step=10)

    # 4. Neural Network Execution
    st.subheader("4. Run Neural Network Training Model")
    if st.button("🚀 Train Multitask Network Model", type="primary"):
        if st.session_state.df is not None and len(st.session_state.df) >= 5:
            with st.spinner("Processing Dataset & Optimizing Weights... Please Wait..."):
                input_cols = ["Power", "Speed", "ExposureTime", "EnergyDensity", "Al", "Fe", "Cr", "Ti", "Si"]
                reg_cols = ["Depth", "Width"]
                cls_col = "DefectType"

                try:
                    X = st.session_state.df[input_cols].values
                    y_reg = st.session_state.df[reg_cols].values.astype("float64")
                    y_cls_raw = st.session_state.df[cls_col].values

                    # Encoding and Normalization pipelines
                    le = LabelEncoder()
                    y_cls_encoded = le.fit_transform(y_cls_raw)

                    num_actual_classes = len(le.classes_)
                    y_cls_onehot = to_categorical(y_cls_encoded, num_classes=num_actual_classes)

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
                        X_scaled, y_reg, y_cls_onehot, test_size=0.2, random_state=42
                    )

                    # Build multi-output network graph: shared trunk, two heads
                    input_layer = Input(shape=(len(input_cols),), name="process_material_input")
                    shared = input_layer
                    for i in range(hidden_layers):
                        shared = Dense(neurons_per_layer, activation="relu", name=f"shared_dense_{i+1}")(shared)

                    # Regression head: predicts Depth & Width
                    reg_branch = Dense(max(neurons_per_layer // 2, 4), activation="relu", name="reg_dense")(shared)
                    reg_output = Dense(len(reg_cols), activation="linear", name="reg_output")(reg_branch)

                    # Classification head: predicts DefectType
                    cls_branch = Dense(max(neurons_per_layer // 2, 4), activation="relu", name="cls_dense")(shared)
                    cls_output = Dense(num_actual_classes, activation="softmax", name="cls_output")(cls_branch)

                    model = Model(inputs=input_layer, outputs=[reg_output, cls_output])
                    model.compile(
                        optimizer="adam",
                        loss={"reg_output": "mse", "cls_output": "categorical_crossentropy"},
                        loss_weights={"reg_output": 1.0, "cls_output": 1.0},
                        metrics={"reg_output": "mae", "cls_output": "accuracy"}
                    )

                    early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

                    history = model.fit(
                        X_train,
                        {"reg_output": y_reg_train, "cls_output": y_cls_train},
                        validation_data=(X_test, {"reg_output": y_reg_test, "cls_output": y_cls_test}),
                        epochs=epochs,
                        batch_size=min(16, max(2, len(X_train) // 4)),
                        callbacks=[early_stop],
                        verbose=0
                    )

                    # Persist trained assets in session state
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.le = le
                    st.session_state.input_cols = input_cols
                    st.session_state.history = history.history

                    st.success(f"Training complete! Stopped after {len(history.history['loss'])} epochs.")

                    # Plot training curves
                    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

                    axes[0].plot(history.history["loss"], label="Train")
                    axes[0].plot(history.history["val_loss"], label="Validation")
                    axes[0].set_title("Total Loss")
                    axes[0].set_xlabel("Epoch")
                    axes[0].legend()

                    axes[1].plot(history.history["reg_output_mae"], label="Train")
                    axes[1].plot(history.history["val_reg_output_mae"], label="Validation")
                    axes[1].set_title("Regression MAE (Depth/Width)")
                    axes[1].set_xlabel("Epoch")
                    axes[1].legend()

                    axes[2].plot(history.history["cls_output_accuracy"], label="Train")
                    axes[2].plot(history.history["val_cls_output_accuracy"], label="Validation")
                    axes[2].set_title("Defect Classification Accuracy")
                    axes[2].set_xlabel("Epoch")
                    axes[2].legend()

                    st.pyplot(fig)

                    # Held-out test metrics
                    test_results = model.evaluate(
                        X_test, {"reg_output": y_reg_test, "cls_output": y_cls_test}, verbose=0
                    )
                    metric_names = model.metrics_names
                    st.subheader("Held-out Test Metrics")
                    st.write({name: round(float(val), 4) for name, val in zip(metric_names, test_results)})

                except KeyError as e:
                    st.error(f"Missing expected column in dataset: {e}. Check the reference structure above.")
                except Exception as e:
                    st.error(f"Training failed: {e}")
        else:
            st.warning("Please load at least 5 rows of data before training.")

# ==============================================================================
# TAB 3: INFERENCE / PREDICTION
# ==============================================================================
with tabs[2]:
    st.header("Predict Melt Pool Geometry & Defect Type")

    if st.session_state.model is None:
        st.warning("No trained model found yet. Go to the *Data Management & Training* tab and train a model first.")
    else:
        st.caption("Enter a new laser process + alloy composition combination to predict melt pool depth, width, and defect type.")

        col1, col2, col3 = st.columns(3)
        with col1:
            power = st.number_input("Power (W)", min_value=0.0, value=200.0, step=5.0)
            speed = st.number_input("Scan Speed (mm/s)", min_value=0.0, value=1000.0, step=10.0)
            exposure = st.number_input("Exposure Time (μs)", min_value=0.0, value=60.0, step=1.0)
        with col2:
            energy_density = st.number_input("Energy Density (J/mm³)", min_value=0.0, value=60.0, step=1.0)
            al = st.number_input("Al (wt.%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1)
            fe = st.number_input("Fe (wt.%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1)
        with col3:
            cr = st.number_input("Cr (wt.%)", min_value=0.0, max_value=100.0, value=2.5, step=0.1)
            ti = st.number_input("Ti (wt.%)", min_value=0.0, max_value=100.0, value=1.5, step=0.1)
            si = st.number_input("Si (wt.%)", min_value=0.0, max_value=100.0, value=1.5, step=0.1)

        if st.button("🔮 Predict", type="primary"):
            input_cols = st.session_state.input_cols
            raw_input = pd.DataFrame(
                [[power, speed, exposure, energy_density, al, fe, cr, ti, si]],
                columns=input_cols
            )
            X_new = st.session_state.scaler.transform(raw_input.values)
            reg_pred, cls_pred = st.session_state.model.predict(X_new, verbose=0)

            depth_pred, width_pred = reg_pred[0]
            defect_idx = np.argmax(cls_pred[0])
            defect_label = st.session_state.le.inverse_transform([defect_idx])[0]
            confidence = cls_pred[0][defect_idx] * 100

            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("Predicted Depth (μm)", f"{depth_pred:.1f}")
            res_col2.metric("Predicted Width (μm)", f"{width_pred:.1f}")
            res_col3.metric("Predicted Defect Type", defect_label, f"{confidence:.1f}% confidence")

            st.subheader("Class Probabilities")
            prob_df = pd.DataFrame({
                "DefectType": st.session_state.le.classes_,
                "Probability": cls_pred[0]
            }).sort_values("Probability", ascending=False)
            st.bar_chart(prob_df.set_index("DefectType"))
