# FraudShield — Credit Card Fraud Detection Working Model

A ready-to-demonstrate interactive ML project based on the uploaded Credit Card Fraud Detection notebook and dataset.

## Quick start

1. Install Python 3.10+ from https://www.python.org/downloads/ (check **Add Python to PATH** on Windows).
2. Unzip the project folder.
3. **Windows:** double-click `START_WINDOWS.bat`. **macOS/Linux:** open Terminal inside this folder, run `bash START_MAC_LINUX.sh`.
4. Your browser should open the dashboard. If not, visit http://localhost:8501.
5. Under **Live Fraud Detector**, pick a genuine sample or fraudulent sample, then click **Detect Fraud**. Switch algorithms to compare.
6. Use **Model Performance** to demonstrate the model evaluation, and **Bulk CSV Prediction** to upload the included samples.

Internet is needed once to install Python packages. Once installed, the app runs locally; no online API or bank access is required.

## What is included

- `app.py`: four-tab interactive dashboard
- `models/fraud_bundle.joblib`: trained Logistic Regression, Random Forest and fitted StandardScaler
- `models/metrics.json`: held-out test evaluation metrics
- `data/demo_transactions.csv`: 16 held-out genuine/fraud sample transactions, with known class labels
- `data/creditcard.csv`: entire provided dataset, used if you want to retrain
- `train_model.py`: retrains both models from original dataset and recreates the sample file
- `requirements.txt`: Python dependencies

## Verified held-out results (80/20 stratified split; random_state=42)

| Model | Accuracy | Precision | Recall | F1 | Fraud caught / 98 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 99.914% | 82.67% | 63.27% | 71.68% | 62 |
| Random Forest | 99.960% | 94.12% | 81.63% | 87.43% | 80 |

There are 284,807 rows, including only 492 known frauds (0.1727%). Fraud-specific recall and precision are more informative than accuracy alone.

## Optional: retrain models

`python train_model.py`

This runs model training using the CSV bundled with the project and refreshes the saved model and evaluation artifacts. You do not need to run it to show the demo.

## Teacher demonstration (2 minutes)

1. Open dashboard; explain dataset size and fraud imbalance.
2. Select **Sample genuine transaction**, click **Detect Fraud**.
3. Select **Sample fraudulent transaction**, click **Detect Fraud**.
4. Switch from **Random Forest** to **Logistic Regression** and compare their outputs.
5. Open **Model Performance**, explain how many actual frauds each algorithm detects and misses.
6. Upload `data/demo_transactions.csv` on the **Bulk CSV Prediction** page.

## Important limitations

- Dataset V1–V28 features are anonymized and cannot be reconstructed from only card number or amount.
- Some fraudulent examples may be missed: classification is statistical, not guaranteed.
- This is an academic prototype, not a live bank transaction integration.

## Streamlit interface edits (teacher presentation)

- The built-in Deploy button, top-right menu, toolbar, header and footer are hidden with `app.py` CSS.
- `.streamlit/config.toml` sets a minimal toolbar mode. The controls are hidden visually, not secured or disabled on the server. Streamlit updates may change CSS selectors.
- The Mac/Linux launcher now installs dependencies inside `.venv`, avoiding the Homebrew externally-managed-environment error. It can be run with `bash START_MAC_LINUX.sh`.
- The Windows launcher also uses `.venv` rather than installing globally.
- In the **Bulk CSV Prediction** tab, upload `data/Fraud_Detection_Upload_Sample.csv` or the labeled CSV; the `Class` column is ignored during predictions. Then download your prediction results as CSV.
