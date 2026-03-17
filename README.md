# Fraud Detection Pipeline

An end-to-end ML pipeline for credit card fraud detection with real-time 
explainability using SHAP.

## Results
- AUC-ROC: 0.9739
- Recall (fraud): 0.88
- Precision (fraud): 0.33
- Dataset: 284,807 transactions, 0.17% fraud rate

## Tech Stack
- XGBoost — gradient boosted classifier
- SMOTE — synthetic minority oversampling
- SHAP — per-transaction explainability
- FastAPI — REST API serving predictions
- scikit-learn — preprocessing and evaluation

## Project Structure
fraud-project/
├── app/
│   └── main.py         # FastAPI endpoint
├── notebooks/
│   └── 01_eda.ipynb    # EDA, training, evaluation
├── requirements.txt
└── README.md

## API Response Example
{
  "fraud_probability": 0.91,
  "is_fraud": true,
  "risk_level": "HIGH",
  "top_factors": [
    {"feature": "V14", "impact": -2.31},
    {"feature": "V11", "impact": -1.87}
  ]
}

## Key Decisions
- Applied SMOTE only to training data to prevent data leakage
- Prioritised recall over precision — missing fraud costs more than 
  false positives
- Used SHAP TreeExplainer for per-transaction explanations, 
  satisfying regulatory explainability requirements