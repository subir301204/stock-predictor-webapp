import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from src.features import create_features

def train(data_path, model_path):
          df = pd.read_csv(data_path, parse_dates=['Date'])
          df = create_features(df)
          X = df[['return', 'SMA5', 'SMA10']]
          y = df['Direction']
          X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
          model = LogisticRegression()
          model.fit(X_train, y_train)
          print(classification_report(y_test, model.predict(X_test)))
          joblib.dump(model, model_path)
