from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from src.features import create_features

def train_rf(data_path, model_path):
          df = pd.read_csv(data_path, parse_dates=['Date'])
          df = create_features(df)
          X = df[['return', 'SMA5', 'SMA10']]
          y = df['Direction']
          model = RandomForestClassifier(n_estimators=100, random_state=42)
          model.fit(X, y)
          joblib.dump(model, model_path)
