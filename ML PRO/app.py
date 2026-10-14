"""Flask REST API serving the tuned Gradient-Boosted model (CO6).  Run: pip install flask joblib scikit-learn pandas && python app.py"""
import json, joblib, pandas as pd
from flask import Flask, request, jsonify
app = Flask(__name__)
model = joblib.load("model.joblib"); meta = json.load(open("meta.json"))
F = meta["features"]
def encode(d):
    gap = d["Unit_Price"] - d["Competitor_Price"]
    row = [d["Unit_Price"], d["Discount_Percent"], int(d["Promotion"] == "Yes"), int(d["Holiday"] == "Yes"),
           d["Marketing_Spend"], d["Competitor_Price"], gap, d["Inventory_Level"], d["Month"], (d["Month"] - 1)//3 + 1,
           meta["days"].index(d["Day_of_Week"]), meta["cats"].index(d["Product_Category"]), meta["regs"].index(d["Region"])]
    return pd.DataFrame([row], columns=F)
@app.post("/predict")
def predict():
    try: return jsonify(predicted_sales=round(float(model.predict(encode(request.get_json()))[0]), 2))
    except (KeyError, ValueError) as e: return jsonify(error=f"Invalid input: {e}"), 400
@app.get("/health")
def health(): return jsonify(status="ok")
@app.get("/metrics")
def metrics(): return jsonify(best_params=meta["best"], test=meta["tuned"], cv_rmse=meta["cv_rmse"])
if __name__ == "__main__": app.run(port=5000)
