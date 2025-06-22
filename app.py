import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from src.features import create_features

st.set_page_config(page_title="📈 Stock Predictor", layout="wide")

st.title("📈 Stock Market Direction Predictor (AI/ML)")
st.write("Upload historical stock data and choose a model to predict whether the stock will go **Up** or **DOWN**.")

# Load CSV 
uploaded_file = st.file_uploader("📥 Upload historical stock data (.csv)", type=["csv"])
if uploaded_file is not None:
          df = pd.read_csv(uploaded_file, parse_dates=['Data'])
else:
          df = pd.read_csv("data/historical.csv", parse_dates=['Date'])
          st.info("Using default data from 'data/historical.csv'.")

# Feature engineering
df = create_features(df)
X = df[['return', 'SMA5', 'SMA10']]
y = df['Direction']

# Select model
model_choice = st.selectbox("🧠 Choose prediction model", ["Logistic Regression", "Random Forest", "LSTM"])

def predict_and_plot(y_true, y_pred):
          fig = go.Figure()
          fig.add_trace(go.Scatter(y=y_true, mode='lines', name='Actual Direction'))
          fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted Direction'))
          fig.update_layout(title='📊 Actual vs Predicted Stock Direction', xaxis_title='Index', yaxis_title='Direction (0 = Down), 1 = Up')
          st.plotly_chart(fig, use_container_width=True)

# Prediction Logic
if st.button("🔮 Predict"):
          if model_choice == "Logistic Regression":
                    model = joblib.load("outputs/models/logistic.pkl")
                    y_pred = model.predict(X)
          
          elif model_choice == "Random Forest":
                    model = joblib.load("outputs/models/rf.pkl")
                    y_pred = model.predict(X)
          
          elif model_choice == "LSTM":
                    model = load_model("outputs/models/lstm.h5")
                    scaler = joblib.load("outputs/models/lstm_scaler.pkl")
                    X_scaled = scaler.transform(X)

                    X_seq = []
                    window = 10
                    for i in range(window, len(X_scaled)):
                              X_seq.append(X_scaled[i-window:i])
                    X_seq = np.array(X_seq)
                    y_pred_raw = model.predict(X_seq)
                    y_pred = (y_pred_raw > 0.5).astype(int).flatten()
                    y = y[window:]
          predict_and_plot(y, y_pred)
          st.success("Prediction completed!")
