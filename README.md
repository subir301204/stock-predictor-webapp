# 📈 Stock Market Direction Predictor (AI/ML Web App)

An AI/ML-powered web application that predicts whether a stock will go **Up** or **Down** using historical price data from Kaggle datasets.

## 🚀 Features

- ✅ Upload your own or use default stock `.csv` data
- ✅ Supports Logistic Regression, Random Forest, and LSTM models
- ✅ Visualizes predictions vs actual values
- ✅ Built with Python, Streamlit, scikit-learn, and Keras
- ✅ Interactive graphs using Plotly

---

## 📂 Project Structure

```
stock-predictor-webapp/
├─ data/
│  └─ historical.csv
├─ src/
│  ├─ features.py
│  ├─ logistic_model.py
│  ├─ rf_model.py
│  ├─ lstm_model.py
├─ outputs/
│  ├─ models/
│  └─ figures/
├─ assets/
├─ app.py
├─ requirements.txt
└─ README.md
```

---

## 📥 Input Format

The app requires a CSV file with the following columns:

- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

Example:

```csv
Date,Open,High,Low,Close,Volume
2023-01-01,140,145,139,143,1000000
```

---

## 🔽 How to Download Kaggle Dataset

1. Install Kaggle CLI:

   ```bash
   pip install kaggle
   ```

2. Place your Kaggle API key (`kaggle.json`) in the correct location:

   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Download a dataset:

   ```bash
   kaggle datasets download -d rohanrao/nifty50-stock-market-data
   unzip nifty50-stock-market-data.zip -d data/
   ```

4. Rename your chosen stock file:

   ```bash
   mv data/RELIANCE.csv data/historical.csv
   ```

---

## 🧠 Models Used

| Model               | Description                              |
| ------------------- | ---------------------------------------- |
| Logistic Regression | Simple linear classifier                 |
| Random Forest       | Tree-based ensemble model                |
| LSTM                | Deep learning model for time-series data |

---

## 📊 Output Format

- Actual vs Predicted direction chart
- Final prediction: `Up (1)` or `Down (0)`
- Rendered using Plotly graphs

---

## ▶️ How to Run

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train models (optional)

```bash
python -c "from src.logistic_model import train; train('data/historical.csv', 'outputs/models/logistic.pkl')"
python -c "from src.rf_model import train_rf; train_rf('data/historical.csv', 'outputs/models/rf.pkl')"
python -c "from src.lstm_model import train_lstm; train_lstm('data/historical.csv', 'outputs/models/lstm.h5')"
```

### Step 3: Run the web app

```bash
streamlit run app.py
```

---

## 🖼️ Screenshots

_Add screenshots here showing:_

- Upload interface
- Historical chart
- Model prediction chart

---

## 🤖 Tech Stack

- Python 3.x
- Streamlit
- scikit-learn
- Pandas, NumPy
- Plotly
- Keras + TensorFlow

---

## 🙋‍♂️ Author

Made by [Subir Ghosh](https://github.com/subir301204)
