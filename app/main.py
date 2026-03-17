from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import shap

app = FastAPI(title="Fraud Detection API")

# Load model and explainer
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
explainer = joblib.load("models/shap_explainer.pkl")

class Transaction(BaseModel):
    features: list[float]  # 29 feature values (V1-V28 + scaled Amount)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(transaction: Transaction):
    X = np.array(transaction.features).reshape(1, -1)
    
    # Prediction
    prob = model.predict_proba(X)[0][1]
    is_fraud = bool(prob > 0.5)
    
    # SHAP explanation
    shap_vals = explainer.shap_values(X)[0]
    
    # Top 5 features driving this prediction
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    top_features = sorted(
        zip(feature_names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]
    
    return {
        "fraud_probability": round(float(prob), 4),
        "is_fraud": is_fraud,
        "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW",
        "top_factors": [
            {"feature": f, "impact": round(float(v), 4)}
            for f, v in top_features
        ]
    }
# ```

# **Then create `app/__init__.py`** — just an empty file. Your folder structure should now look like:
# ```
# fraud-project/
# ├── app/
# │   ├── __init__.py
# │   └── main.py
# ├── data/
# ├── models/
# │   ├── fraud_model.pkl
# │   ├── scaler.pkl
# │   └── shap_explainer.pkl
# └── notebooks/