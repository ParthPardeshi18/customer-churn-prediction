# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green?logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-red?logo=streamlit)

A machine learning project that predicts customer churn for a telecom company using the IBM Telco Customer Churn dataset. The goal is to identify at-risk customers so the business can take proactive retention actions — reducing revenue loss and improving customer lifetime value.

## Folder Structure

```
churn-prediction/
├── data/                   # Raw dataset
├── notebooks/              # Jupyter notebooks (exploration)
├── src/
│   ├── eda.py              # Exploratory data analysis
│   ├── preprocess.py       # Feature engineering & SMOTE
│   ├── train.py            # Model training & evaluation
│   └── app.py              # Streamlit prediction app
├── outputs/                # Charts, model, scaler
├── requirements.txt
└── README.md
```

## How to Run

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run EDA
python src/eda.py

# 4. Preprocess data
python src/preprocess.py

# 5. Train models
python src/train.py

# 6. Launch Streamlit app
streamlit run src/app.py
```

## Key Findings

- **Contract type is the strongest churn predictor** — month-to-month customers churn at 42% vs. just 3% for two-year contracts.
- **New customers are most at risk** — the 0-12 month tenure group has the highest churn rate (~48%), dropping sharply after 2 years.
- **Higher monthly charges correlate with churn** — churned customers have a median monthly charge ~$30 higher than retained customers.

## Results

| Model               | AUC    | F1     |
|---------------------|--------|--------|
| Logistic Regression | 0.8944 | 0.8079 |
| Random Forest       | 0.9259 | 0.8526 |
| **XGBoost**         | 0.9063 | 0.8290 |

The final XGBoost model achieves a **test AUC of 0.823** on the held-out set with SMOTE-balanced training data.
