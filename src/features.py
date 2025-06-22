import pandas as pd

def create_features(df):
          df['return'] = df['Close'].pct_change()
          df['SMA5'] = df['Close'].rolling(5).mean()
          df['SAM10'] = df['Close'].rolling(10).mean()
          df['Direction'] = (df['return'].shift(-1) > 0).astype(int)
          return df.dropna()
