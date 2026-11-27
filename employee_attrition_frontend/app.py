from flask import Flask, render_template, request, send_file, flash, redirect, url_for
import pandas as pd
import joblib
import io
import os

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "attrition_model.joblib")

# The 30 predictor columns used by the supplied notebook.
FIELDS = [
    {"name":"Age","label":"Age","type":"number","min":18,"max":65,"default":35},
    {"name":"BusinessTravel","label":"Business travel","type":"select","options":["Travel_Rarely","Travel_Frequently","Non-Travel"]},
    {"name":"DailyRate","label":"Daily rate","type":"number","min":100,"max":1500,"default":800},
    {"name":"Department","label":"Department","type":"select","options":["Research & Development","Sales","Human Resources"]},
    {"name":"DistanceFromHome","label":"Distance from home","type":"number","min":1,"max":29,"default":5},
    {"name":"Education","label":"Education level","type":"select","options":["1","2","3","4","5"]},
    {"name":"EducationField","label":"Education field","type":"select","options":["Life Sciences","Medical","Marketing","Technical Degree","Other","Human Resources"]},
    {"name":"EnvironmentSatisfaction","label":"Environment satisfaction","type":"select","options":["1","2","3","4"]},
    {"name":"Gender","label":"Gender","type":"select","options":["Male","Female"]},
    {"name":"HourlyRate","label":"Hourly rate","type":"number","min":30,"max":100,"default":65},
    {"name":"JobInvolvement","label":"Job involvement","type":"select","options":["1","2","3","4"]},
    {"name":"JobLevel","label":"Job level","type":"select","options":["1","2","3","4","5"]},
    {"name":"JobRole","label":"Job role","type":"select","options":["Sales Executive","Research Scientist","Laboratory Technician","Manufacturing Director","Healthcare Representative","Manager","Sales Representative","Research Director","Human Resources"]},
    {"name":"JobSatisfaction","label":"Job satisfaction","type":"select","options":["1","2","3","4"]},
    {"name":"MaritalStatus","label":"Marital status","type":"select","options":["Single","Married","Divorced"]},
    {"name":"MonthlyIncome","label":"Monthly income","type":"number","min":1000,"max":20000,"default":5000},
    {"name":"MonthlyRate","label":"Monthly rate","type":"number","min":2000,"max":27000,"default":14000},
    {"name":"NumCompaniesWorked","label":"Companies worked at","type":"number","min":0,"max":9,"default":2},
    {"name":"OverTime","label":"Overtime","type":"select","options":["No","Yes"]},
    {"name":"PercentSalaryHike","label":"Salary hike (%)","type":"number","min":11,"max":25,"default":15},
    {"name":"PerformanceRating","label":"Performance rating","type":"select","options":["3","4"]},
    {"name":"RelationshipSatisfaction","label":"Relationship satisfaction","type":"select","options":["1","2","3","4"]},
    {"name":"StockOptionLevel","label":"Stock option level","type":"select","options":["0","1","2","3"]},
    {"name":"TotalWorkingYears","label":"Total working years","type":"number","min":0,"max":40,"default":10},
    {"name":"TrainingTimesLastYear","label":"Training times last year","type":"number","min":0,"max":6,"default":3},
    {"name":"WorkLifeBalance","label":"Work-life balance","type":"select","options":["1","2","3","4"]},
    {"name":"YearsAtCompany","label":"Years at company","type":"number","min":0,"max":40,"default":5},
    {"name":"YearsInCurrentRole","label":"Years in current role","type":"number","min":0,"max":18,"default":3},
    {"name":"YearsSinceLastPromotion","label":"Years since last promotion","type":"number","min":0,"max":15,"default":1},
    {"name":"YearsWithCurrManager","label":"Years with current manager","type":"number","min":0,"max":17,"default":3},
]

def load_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def score_frame(frame, bundle):
    columns = bundle["columns"]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))
    clean = frame[columns].copy()
    probabilities = bundle["model"].predict_proba(clean)[:, 1]
    threshold = float(bundle["threshold"])
    result = frame.copy()
    result["attrition_probability"] = probabilities
    result["risk_level"] = pd.cut(probabilities, bins=[-0.001, .35, .65, 1.001],
                                  labels=["Low", "Moderate", "High"])
    result["predicted_to_leave"] = (probabilities >= threshold).astype(int)
    result["prediction"] = result["predicted_to_leave"].map({0:"Stay", 1:"Leave"})
    return result

@app.route("/")
def index():
    bundle = load_bundle()
    return render_template("index.html", fields=FIELDS, model_ready=bundle is not None)

@app.route("/predict", methods=["POST"])
def predict():
    bundle = load_bundle()
    if bundle is None:
        flash("Model file not found. Run your Colab notebook through the model-saving cell, then place attrition_model.joblib beside app.py.", "error")
        return redirect(url_for("index"))
    try:
        row = {}
        for field in FIELDS:
            raw = request.form.get(field["name"], "")
            row[field["name"]] = int(raw) if field["type"] == "number" or field["name"] in {
                "Education","EnvironmentSatisfaction","JobInvolvement","JobLevel","JobSatisfaction",
                "PerformanceRating","RelationshipSatisfaction","StockOptionLevel","WorkLifeBalance"
            } else raw
        result = score_frame(pd.DataFrame([row]), bundle).iloc[0]
        prob = float(result["attrition_probability"])
        return render_template("result.html", result=result, probability=prob,
                               threshold=float(bundle["threshold"]),
                               risk=str(result["risk_level"]))
    except Exception as exc:
        flash(f"Prediction could not be completed: {exc}", "error")
        return redirect(url_for("index"))

@app.route("/bulk", methods=["POST"])
def bulk():
    bundle = load_bundle()
    if bundle is None:
        flash("Model file not found. Add attrition_model.joblib beside app.py first.", "error")
        return redirect(url_for("index"))
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename.lower().endswith(".csv"):
        flash("Please upload a CSV file.", "error")
        return redirect(url_for("index"))
    try:
        data = pd.read_csv(uploaded)
        scored = score_frame(data, bundle)
        csv_bytes = scored.to_csv(index=False).encode("utf-8")
        buffer = io.BytesIO(csv_bytes)
        buffer.seek(0)
        summary = {
            "total": len(scored),
            "high": int((scored["risk_level"] == "High").sum()),
            "moderate": int((scored["risk_level"] == "Moderate").sum()),
            "low": int((scored["risk_level"] == "Low").sum()),
            "predicted": int(scored["predicted_to_leave"].sum())
        }
        table = scored.head(100).copy()
        table["attrition_probability"] = (table["attrition_probability"] * 100).round(1).astype(str) + "%"
        return render_template("bulk_result.html", summary=summary,
                               tables=[table.to_html(classes="data-table", index=False, border=0)],
                               download_data=buffer.getvalue().decode("utf-8"))
    except Exception as exc:
        flash(f"Could not score this file: {exc}", "error")
        return redirect(url_for("index"))

@app.route("/download-results", methods=["POST"])
def download_results():
    content = request.form.get("csv_data", "")
    return send_file(io.BytesIO(content.encode("utf-8")), mimetype="text/csv",
                     as_attachment=True, download_name="employee_attrition_predictions.csv")

if __name__ == "__main__":
    app.run(debug=True)
