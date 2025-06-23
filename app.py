import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from keras.models import load_model
import os
from src.features import create_features

st.set_page_config(page_title="📈 Stock Predictor", layout="wide")

st.title("📈 Stock Market Direction Predictor (AI/ML)")
st.write("Upload historical stock data and choose a model to predict whether the stock will go **Up** or **DOWN**.")

MODEL_PATHS = {
    "Logistic Regression": "outputs/models/logistic.pkl",
    "Random Forest": "outputs/models/rf.pkl",
    "LSTM": "outputs/models/lstm.h5",
    "LSTM_SCALER": "outputs/models/lstm_scaler.pkl"
}

@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded file or a default path."""
    if uploaded_file is not None:
        try:
            # Corrected 'Data' to 'Date' to match default and common usage
            df = pd.read_csv(uploaded_file, parse_dates=['Date'])
        except (ValueError, KeyError):
            st.error("The uploaded CSV must contain a 'Date' column. Please check the file.")
            return None
    else:
        default_path = "data/historical.csv"
        if not os.path.exists(default_path):
            st.error(f"Default data file not found at '{default_path}'. Please upload a file.")
            return None
        df = pd.read_csv(default_path, parse_dates=['Date'])
        st.info(f"Using default data from '{default_path}'.")
    return df

def get_predictions(model_choice, X, y):
    """Loads the selected model and returns predictions."""
    try:
        if model_choice == "Logistic Regression":
            model = joblib.load(MODEL_PATHS["Logistic Regression"])
            y_pred = model.predict(X)
            return y, y_pred
        
        elif model_choice == "Random Forest":
            model = joblib.load(MODEL_PATHS["Random Forest"])
            y_pred = model.predict(X)
            return y, y_pred
        
        elif model_choice == "LSTM":
            model = load_model(MODEL_PATHS["LSTM"])
            scaler = joblib.load(MODEL_PATHS["LSTM_SCALER"])
            X_scaled = scaler.transform(X)

            X_seq = []
            window = 10
            for i in range(window, len(X_scaled)):
                X_seq.append(X_scaled[i-window:i])
            X_seq = np.array(X_seq)
            
            y_pred_raw = model.predict(X_seq)
            y_pred = (y_pred_raw > 0.5).astype(int).flatten()
            # Align y_true with the predictions by removing the window period
            y_true = y.iloc[window:]
            return y_true, y_pred
    except FileNotFoundError as e:
        st.error(f"Model file not found: `{e.filename}`. Please ensure you have trained the models.")
        st.info("To train the models, run the scripts in the `src/` directory, for example:")
        st.code("python src/logistic_model.py")
        st.info("Also, ensure you have downloaded the data by running `python src/download_kaggle.py` and renamed the CSV to `data/historical.csv`.")
        return None, None

def plot_predictions(y_true, y_pred):
    """Plots the actual vs. predicted directions as line graphs with zoom support."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true.index, y=y_true,
        mode='lines',
        name='Actual Direction',
        line=dict(color='royalblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=y_true.index, y=y_pred,
        mode='lines',
        name='Predicted Direction',
        line=dict(color='firebrick', width=2, dash='dash')
    ))
    fig.update_layout(
        title='📊 Actual vs Predicted Stock Direction',
        xaxis_title='Date',
        yaxis_title='Direction (0 = Down, 1 = Up)',
        xaxis_rangeslider_visible=True  # Enable zoom/scroll
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_historical_data(df):
    """Plots the historical closing price of the stock as a line graph."""
    st.subheader("Historical Data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='cyan')))
    fig.update_layout(
        title='📈 Historical Stock Closing Price',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig, use_container_width=True)

uploaded_file = st.file_uploader("📥 Upload historical stock data (.csv)", type=["csv"])
raw_df = load_data(uploaded_file)

if raw_df is not None:
    plot_historical_data(raw_df)

    st.subheader("Model Prediction")
    processed_df = create_features(raw_df.copy())
    X = processed_df[['return', 'SMA5', 'SMA10']]
    y = processed_df['Direction']
    model_choice = st.selectbox("🧠 Choose prediction model", ["Logistic Regression", "Random Forest", "LSTM"])

    if st.button("🔮 Predict"):
        with st.spinner("Predicting..."):
            y_true, y_pred = get_predictions(model_choice, X, y)
        if y_true is not None and y_pred is not None:
            plot_predictions(y_true, y_pred)
            # Calculate prediction accuracy
            accuracy = (y_true.values == y_pred).mean()
            st.success(f"Prediction Accuracy: {accuracy:.2%}")
            # Show last prediction direction
            last_pred = y_pred[-1]
            direction = "Up 📈" if last_pred == 1 else "Down 📉"
            st.info(f"The model predicts the next direction will be: **{direction}**")
