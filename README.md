# 📈 Stock Market Direction Predictor (AI/ML Web App)

A machine learrning web application that predicts whether a stock will go **Up** or **Down** using historical price data from Kaggle datasets.

## 🚀 Features

- ✅ Upload or use Kuggle stock `.csv` data
- ✅ Supports Logistic Regression, Random Forest, and LSTM models
- ✅ Shows prediction vs actual direction (graph)
- ✅ Built using Python, Streamlit, and scikit-learn
- ✅ Optional Flask support

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
│  └─ app.py
├─ templates/
│  └─ index.html
├─ assets/
├─ outputs/
│  ├─ models/
│  └─ figures/
├─ requirements.txt
└─ README.md
```

---

## 📥 Input Format

- Based on a Kaggle dataset with columns:
  - `Data`, `Open`, `High`, `Low`, `Close`, `Volume`

Example (from Kaggle CSV):

```csv
Data, Open, High, Low, Close, Volume
2023-01-01, 140, 145, 139, 143, 1000000
```

---

## 🔽 How to Download Kagale Dataset

1. Install the Kaggle APP:

   ```bash
   pip install kaggle
   ```

2. Upload your `kaggle.json` (from https://www.kaggle.com/account) to `~/.kaggle/`:

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

4. Choose a file (e.g., `RELIANCE.csv`) and rename to:

   ```bash
   mv data/RELIANCE.csv data/historical.csv
   ```

---

## 🧠 Models Used

| Model               | Description                              |
| ------------------- | ---------------------------------------- |
| Logistic Regression | Basic linear classifier                  |
| Random Forest       | Tree-based ensemble model                |
| LSTM (optional)     | Deep learning for sequential time-series |

---

## 📊 Output Format

- Displays **Actual** vs **Predicted** directions
- Final prediction: `Up (1)` or `Down (0)`
- Graphs rendered using **Plotly**

---

## ▶️ How to Run

### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 2: Train the Models (Optional)

```bash
python -c "from src.logistic_model import traain; train('data/historical.csv', 'outputs/models/logistic.pkl')"
python -c "from src.rf_model import train_rf; train_rf('data/historical.csv', 'outputs/models/rf.pkl')"
python -c "from src.lstm_model import train_lstm; train_lstm('data/historical.csv', 'outputs/models/lstm.h5')"
```

### Step 3: Run the Web App

```bash
streamlit run app.py
```

---

## 📷 Screenshorts

_I have to add some screenshort here of the app UI, graph view, and predictions._

## 🤖 Tech Stack

- Python 3.x
- scikit-learn
- Streamlit
- Pandas, NumPy
- Plotly
- (Optional) Keras + TensorFlow for LSTM

---

## 🙋‍♂️ Author

Made by [Subir Ghosh](https://github.com/subir301204)

---
