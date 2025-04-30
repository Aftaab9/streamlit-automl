import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocessing, model_selector, evaluator, model_persistence, visualizer

st.set_page_config(page_title="AutoML Dashboard", layout="wide")
st.title("📊 Automated Machine Learning Dashboard")

# Load dataset
st.sidebar.header("1. Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    with st.expander("Data Summary"):
        st.write(df.describe())
        st.write("Missing values:")
        st.write(df.isnull().sum())

    # Preprocessing
    st.sidebar.header("2. Preprocessing")
    target = st.sidebar.selectbox("Select Target Variable", df.columns)
    strategy = st.sidebar.selectbox("Missing Value Strategy", ["drop", "mean", "median", "most_frequent"])
    
    df_cleaned = preprocessing.clean_data(df.copy(), strategy)
    X_train, X_test, y_train, y_test = preprocessing.split_data(df_cleaned, target)

    st.subheader("Preprocessed Data")
    st.write(f"✅ Data cleaned using '{strategy}' strategy. Splitting 80/20 into train/test sets.")
    st.write("Train shape:", X_train.shape)
    st.write("Test shape:", X_test.shape)

    # Model selection and training
    st.sidebar.header("3. Train a Model")
    task_type = st.sidebar.radio("Prediction Task", ["Classification", "Regression"])
    models = model_selector.get_models(task_type)
    selected_model = st.sidebar.selectbox("Select a Model", list(models.keys()))

    if st.sidebar.button("Train Model"):
        model = models[selected_model]
        model.fit(X_train, y_train)
        st.success(f"{selected_model} trained successfully!")

        # Evaluation
        st.subheader("Model Evaluation")
        metrics = evaluator.evaluate_model(model, X_test, y_test, task_type)
        for k, v in metrics.items():
            st.write(f"**{k}**: {v:.4f}")

        # Save model
        if st.button("💾 Save Model"):
            model_persistence.save_model(model, f"{selected_model.lower()}_model.pkl")
            st.success(f"Model saved as '{selected_model.lower()}_model.pkl'.")

        # Visualizations
        st.subheader("Visualization")
        if task_type == "Classification":
            fig = visualizer.plot_confusion_matrix(model, X_test, y_test)
            st.pyplot(fig)
        else:
            fig = visualizer.plot_prediction_scatter(model, X_test, y_test)
            st.pyplot(fig)

# Load model section
st.sidebar.header("4. Load Saved Model")
load_model = st.sidebar.file_uploader("Upload a Saved Model (.pkl)", type=["pkl"])
if load_model:
    loaded_model = model_persistence.load_model(load_model)
    st.success("Model loaded successfully.")
    if df is not None and target in df.columns:
        _, X_test_loaded, _, y_test_loaded = preprocessing.split_data(df.copy(), target)
        st.subheader("Evaluation of Loaded Model")
        loaded_metrics = evaluator.evaluate_model(loaded_model, X_test_loaded, y_test_loaded, task_type)
        for k, v in loaded_metrics.items():
            st.write(f"**{k}**: {v:.4f}")
