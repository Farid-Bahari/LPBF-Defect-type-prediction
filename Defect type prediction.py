import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import streamlit as st
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- Streamlit App ---
st.title("Melt Pool Prediction App")
st.write("Hybrid FEM + ML model for predicting melt pool dimensions and defect type.")

# --- Dataset initialization with adjustable rows ---
st.sidebar.subheader("Dataset Settings")
n_samples = st.sidebar.number_input("Number of rows", min_value=10, max_value=500, value=10, step=10)

if "df" not in st.session_state or len(st.session_state.df) != n_samples:
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
        "Depth": np.random.uniform(0, 2000, n_samples),
        "Width": np.random.uniform(0, 3000, n_samples),
        "DefectType": np.random.choice(["LoF", "Keyhole", "Full-dense"], n_samples)
    })

# --- Editable table ---
st.subheader("Dataset Editor")
edited_df = st.data_editor(st.session_state.df, num_rows="dynamic")
st.session_state.df = edited_df  # update stored dataset

# --- Train Model ---
input_cols = [ "Power", "Speed", "ExposureTime", "EnergyDensity","Al", "Fe", "Cr", "Ti", "Si"]
reg_cols = ["Depth", "Width"]
cls_col = "DefectType"

X = st.session_state.df[input_cols].values
y_reg = st.session_state.df[reg_cols].values.astype("float64")
y_cls_raw = st.session_state.df[cls_col].values

# Encode defect type
le = LabelEncoder()
y_cls_encoded = le.fit_transform(y_cls_raw)
y_cls_onehot = to_categorical(y_cls_encoded, num_classes=3)

# Normalize inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X_scaled, y_reg, y_cls_onehot, test_size=0.2, random_state=42
)

# --- Model parameters ---
st.sidebar.subheader("Model Parameters")
hidden_layers = st.sidebar.slider("Number of hidden layers", 1, 5, 2, step=1)
neurons_per_layer = st.sidebar.slider("Neurons per hidden layer", 4, 128, 4, step=4)
epochs = st.sidebar.slider("Epochs", 10, 500, 100, step=10)

if st.sidebar.button("Train Model"):
    # Fix seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # --- Define multitask model ---
    input_layer = Input(shape=(9,))   # fixed at 9
    shared = input_layer

    # add variable number of hidden layers
    for _ in range(hidden_layers):
        shared = Dense(neurons_per_layer, activation='relu')(shared)

    reg_output = Dense(2, activation='relu', name='regression')(shared)
    cls_output = Dense(3, activation='softmax', name='classification')(shared)

    model = Model(inputs=input_layer, outputs=[reg_output, cls_output])
    model.compile(
        optimizer='adam',
        loss={'regression': 'mse', 'classification': 'categorical_crossentropy'},
        metrics={'regression': 'mae', 'classification': 'accuracy'}
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

    history = model.fit(
        X_train,
        {'regression': y_reg_train, 'classification': y_cls_train},
        validation_data=(X_test, {'regression': y_reg_test, 'classification': y_cls_test}),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=0
    )

    # Save trained objects in session_state
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.le = le

    st.success("Model training completed!")

    # --- Plot training curves ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['regression_mae'], label="Train MAE")
    ax[0].plot(history.history['val_regression_mae'], label="Val MAE")
    ax[0].set_title("Regression MAE")
    ax[0].legend()

    ax[1].plot(history.history['classification_accuracy'], label="Train Acc")
    ax[1].plot(history.history['val_classification_accuracy'], label="Val Acc")
    ax[1].set_title("Classification Accuracy")
    ax[1].legend()

    st.pyplot(fig)

# --- Predict new condition ---
if st.session_state.model is not None:
    st.header("Predict New Condition")

    col1, col2, col3 = st.columns(3)
    with col1:
        al = st.number_input("Al [%]", 0.0, 100.0, 94.5, step=0.1)
        fe = st.number_input("Fe [%]", 0.0, 100.0, 1.8, step=0.1)
        cr = st.number_input("Cr [%]", 0.0, 100.0, 1.4, step=0.1)
    with col2:
        ti = st.number_input("Ti [%]", 0.0, 100.0, 1.2, step=0.1)
        si = st.number_input("Si [%]", 0.0, 100.0, 1.1, step=0.1)
        power = st.number_input("Laser Power [W]", 100, 1000, 200)
    with col3:
        speed = st.number_input("Scan Speed [mm/s]", 100, 2000, 842)
        et = st.number_input("Exposure Time [µs]", 10, 200, 85)
        ed = st.number_input("Energy Density [J/mm³]", 10, 200, 73)

    if st.button("Predict"):
        X_new = np.array([[al, fe, cr, ti, si, power, speed, et, ed]])
        X_new_scaled = st.session_state.scaler.transform(X_new)

        y_reg_pred, y_cls_pred = st.session_state.model.predict(X_new_scaled)
        defect = st.session_state.le.inverse_transform([np.argmax(y_cls_pred)])

        st.subheader("Prediction Results")
        st.write("**Melt Pool Depth:**", float(y_reg_pred[0, 0]))
        st.write("**Melt Pool Width:**", float(y_reg_pred[0, 1]))
        st.write("**Defect Type:**", defect[0])
