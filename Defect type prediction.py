# ==============================================================================
# DATA MANAGEMENT & TRAINING
# ==============================================================================

st.subheader("2. Dataset")

data_option = st.radio(
    "Choose your dataset:",
    [
        "🧪 Use Example Dataset",
        "📁 Upload CSV",
        "✏️ Edit / Paste Data",
        "🎲 Generate Synthetic Data"
    ],
    horizontal=True
)

# ----------------------------------------------------------------------
# Example dataset
# ----------------------------------------------------------------------

if data_option == "🧪 Use Example Dataset":

    example_path = "data/example_dataset.csv"

    if os.path.exists(example_path):

        example_df = pd.read_csv(example_path)

        st.session_state.df = example_df

        st.success(
            f"Example dataset loaded successfully: "
            f"{len(example_df)} rows × {len(example_df.columns)} columns"
        )

    else:

        st.error(
            "Example dataset was not found. "
            "Make sure 'data/example_dataset.csv' exists in your repository."
        )


# ----------------------------------------------------------------------
# Upload user's own dataset
# ----------------------------------------------------------------------

elif data_option == "📁 Upload CSV":

    uploaded_file = st.file_uploader(
        "Upload your experimental or FEM dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        uploaded_df = pd.read_csv(uploaded_file)

        required_cols = [
            "Power",
            "Speed",
            "ExposureTime",
            "EnergyDensity",
            "Al",
            "Fe",
            "Cr",
            "Ti",
            "Si",
            "Depth",
            "Width",
            "DefectType"
        ]

        missing_cols = [
            col for col in required_cols
            if col not in uploaded_df.columns
        ]

        if not missing_cols:

            st.session_state.df = uploaded_df

            st.success(
                f"Dataset loaded successfully: "
                f"{len(uploaded_df)} rows"
            )

        else:

            st.error(
                f"Missing columns: {missing_cols}"
            )


# ----------------------------------------------------------------------
# Manual editing
# ----------------------------------------------------------------------

elif data_option == "✏️ Edit / Paste Data":

    if st.session_state.df is None:

        st.session_state.df = template_df.copy()


# ----------------------------------------------------------------------
# Synthetic dataset
# ----------------------------------------------------------------------

elif data_option == "🎲 Generate Synthetic Data":

    n_samples = st.sidebar.number_input(
        "Number of synthetic rows",
        min_value=10,
        max_value=500,
        value=50,
        step=10
    )

    if st.button("Generate Synthetic Dataset"):

        np.random.seed(42)

        st.session_state.df = pd.DataFrame({

            "Power":
                np.random.randint(100, 1000, n_samples),

            "Speed":
                np.random.randint(100, 2000, n_samples),

            "ExposureTime":
                np.random.randint(10, 200, n_samples),

            "EnergyDensity":
                np.random.randint(10, 200, n_samples),

            "Al":
                np.random.uniform(85, 95, n_samples),

            "Fe":
                np.random.uniform(2, 8, n_samples),

            "Cr":
                np.random.uniform(1, 5, n_samples),

            "Ti":
                np.random.uniform(0, 3, n_samples),

            "Si":
                np.random.uniform(0.5, 3, n_samples),

            "Depth":
                np.random.uniform(10, 2000, n_samples),

            "Width":
                np.random.uniform(50, 3000, n_samples),

            "DefectType":
                np.random.choice(
                    ["LoF", "Keyhole", "Full-dense"],
                    n_samples
                )
        })

        st.success("Synthetic dataset generated.")


# ----------------------------------------------------------------------
# Show active dataset
# ----------------------------------------------------------------------

if st.session_state.df is not None:

    st.subheader("3. Active Working Dataset")

    st.caption(
        "Review or edit the dataset before training. "
        "You can modify individual cells or add/remove rows."
    )

    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        height=400
    )

    st.session_state.df = edited_df
