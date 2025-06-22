import numpy as np
import pandas as pd
import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from src.features import create_features

def train_lstm(data_path, model_path):
          df = pd.read_csv(data_path, parse_dates=['Date'])
          df = create_features(df)
          X = df[['return', 'SMA5', 'SMA10']].values
          y = df['Direction'].values

          scaler = MinMaxScaler()
          X_scaled = scaler.fit_transform(X)

          X_seq, y_seq = [], []
          window = 10
          for i in range(window, len(X_scaled)):
                    X_seq.append(X_scaled[i-window:i])
                    y_seq.append(y[i])
          
          X_seq, y_seq = np.array(X_seq), np.array(y_seq)

          model = Sequential()
          model.add(LSTM(50, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2])))
          model.add(Dense(1, activation='sigmoid'))
          model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

          model.fit(X_seq, y_seq, epochs=10, batch_size=16, verbose=1)
          model.save(model_path)
          joblib.dump(scaler, model_path.replace('.h5', '_scaler.pkl'))
