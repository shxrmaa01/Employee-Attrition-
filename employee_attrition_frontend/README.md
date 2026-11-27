# PeoplePulse: Employee Attrition Prediction Frontend

A Flask web interface for the employee attrition model in the supplied Google Colab notebook.

## Features
- Single-employee prediction form with all 30 predictor fields expected by the notebook.
- CSV bulk scoring and downloadable results.
- Probability score, risk band, and model decision (`Stay` / `Leave`).
- Responsive HR analytics dashboard.

## Important: add your trained model
The notebook saves a file named `attrition_model.joblib` in cell 15. Run the notebook through that cell, then download the file from Colab and place it in this project folder beside `app.py`.

The web app does not retrain the model. It loads the exact saved model bundle from your notebook:
- `model`
- `threshold`
- `columns`

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Put `attrition_model.joblib` beside `app.py`.
6. Start the app:

   ```bash
   python app.py
   ```

7. Open http://127.0.0.1:5000

## CSV upload format
The CSV must contain the 30 predictor columns used by the notebook. Use the original IBM HR Employee Attrition dataset's predictor column names and spelling. Do not include `Attrition` as a required input. Extra columns are retained in the output but ignored by the model.

## Notebook evaluation metrics shown on the dashboard
These are copied from the supplied notebook's held-out test output: ROC-AUC 0.812, PR-AUC 0.588, accuracy approximately 0.85, and the top 10% highest-risk group contained 40% of test-set leavers. They describe the notebook's evaluation, not a guarantee of live prediction performance.

## Responsible use
Employee attrition scores are uncertain model estimates and can reflect bias in historical data. Use them only as decision support, validate performance on relevant current data, protect employee data, and never use a score as the sole basis for employment decisions.
