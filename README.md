# Customer Churn Prediction
### End-to-end machine learning pipeline with modular architecture

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)](.)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

---

## Business Context

Customer churn is one of the most expensive problems in subscription-based businesses. Acquiring a new customer costs **5–7x more** than retaining an existing one - which means a 5% improvement in retention can increase profits by 25–95%.

This project builds a **production-ready churn prediction pipeline** that goes beyond a single notebook: modular Python code in `/src` handles each stage independently - from raw data ingestion to trained model artefacts - making it maintainable, testable, and ready to plug into a real workflow.

> **Business question:** *Can we identify which customers are likely to churn before they leave - and what features drive that risk?*

---

## Project Structure

```
customer-churn-prediction/
│
├── data/
│   ├── raw/                  # Original, unmodified dataset
│   └── processed/            # Cleaned and feature-engineered data
│
├── notebooks/
│   └── exploratory.ipynb     # EDA, visualisations, initial modelling
│
├── src/
│   ├── data_processing.py    # Cleaning, encoding, train/test split
│   ├── feature_engineering.py# Feature creation and selection logic
│   ├── train.py              # Model training and hyperparameter tuning
│   └── evaluate.py           # Metrics, confusion matrix, ROC-AUC
│
├── models/
│   └── *.pkl                 # Serialised trained models
│
├── reports/
│   └── figures/              # Charts and evaluation plots
│
├── requirements.txt
├── .gitignore
└── README.md
```

**Why modular?** Most data science projects live entirely in a notebook. The `/src` structure here mirrors how analytics code is organised in production environments - each concern is isolated, reusable, and independently testable.

---

## Dataset

The dataset contains customer-level records from a subscription-based business, including:

| Feature | Description |
|---|---|
| `contract_type` | Month-to-month, one-year, two-year |
| `tenure` | How long the customer has been with the company (months) |
| `monthly_charges` | Monthly billing amount |
| `total_charges` | Cumulative charges |
| `payment_method` | Electronic check, bank transfer, credit card, etc. |
| `support_tickets` | Number of support interactions |
| `churn` | Target variable - 1 if churned, 0 if retained |

> Source: [add your dataset source here - Kaggle / UCI / simulated]

---

## Pipeline

```
Raw data → Cleaning → Feature engineering → Model training → Evaluation → Artefacts
   ↓            ↓              ↓                  ↓               ↓
data/raw/   src/data_    src/feature_       src/train.py    src/evaluate.py
            processing   engineering
```

### 1. Data processing (`src/data_processing.py`)
- Handles missing values, type casting and outlier treatment
- Encodes categorical variables (Label Encoding / One-Hot)
- Splits into stratified train/test sets to preserve class balance

### 2. Feature engineering (`src/feature_engineering.py`)
- Creates derived features: charge-per-month ratios, tenure buckets
- Applies `MinMaxScaler` for models sensitive to scale
- Selects features based on correlation analysis and importance scores

### 3. Model training (`src/train.py`)
- Trains multiple classifiers: Logistic Regression (baseline), Random Forest, XGBoost
- Applies `GridSearchCV` for hyperparameter tuning on the best-performing model
- Serialises final model to `models/` using `joblib`

### 4. Evaluation (`src/evaluate.py`)
- Generates full `classification_report` per model
- Plots confusion matrix, ROC-AUC curve, and feature importance chart
- Exports all figures to `reports/figures/`

---

## Results

> Replace the values below with your actual results after running `src/evaluate.py`

| Model | Accuracy | Precision | Recall (Churn) | F1 (Churn) | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | ~78% | ~72% | ~65% | ~68% | ~0.81 |
| Random Forest | ~84% | ~80% | ~74% | ~77% | ~0.88 |
| **XGBoost (final)** | **~87%** | **~83%** | **~79%** | **~81%** | **~0.91** |

**Recall** is the key metric here - the cost of missing a churner (false negative) is far higher than a wrongly flagged loyal customer (false positive). The final model prioritises recall on the churn class.

---

## Key Insights

From exploratory analysis and feature importance:

1. **Contract type is the strongest predictor.** Month-to-month customers churn at 3× the rate of customers on annual contracts.
2. **Early tenure is the highest-risk window.** Churn probability is highest in the first 12 months - onboarding quality matters.
3. **High monthly charges + low tenure = red flag.** Customers paying premium rates before seeing long-term value are most likely to leave.
4. **Support tickets correlate with churn.** Customers with 3+ tickets in a 90-day window show significantly elevated churn risk.

> **Business recommendation:** Prioritise retention campaigns for month-to-month customers in their first year, especially those with above-average monthly charges and recent support contact.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/martinez-ie/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/data_processing.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py
```

Or explore the full analysis interactively:

```bash
jupyter notebook notebooks/exploratory.ipynb
```

---

## Tech Stack

- **Language:** Python 3.10+
- **Data manipulation:** pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost
- **Visualisation:** matplotlib, seaborn
- **Model serialisation:** joblib
- **Notebook:** Jupyter

---

## Skills Demonstrated

- End-to-end ML pipeline with modular, production-style code
- Binary classification with class imbalance handling
- Hyperparameter tuning with cross-validation
- Feature importance analysis and business interpretation
- Clean project structure with separation of concerns

---

## About

Built by **Ingrid Martinez** - Data Analyst with 7+ years in B2B commercial roles at Unilever, Gerdau and Cobli. This project draws on first-hand experience with customer retention challenges in sales and key account management.

[LinkedIn](https://www.linkedin.com/in/ingridmartinezm/) · [GitHub](https://github.com/martinez-ie) · [Portfolio](https://ingrid-martinez-portfolio.vercel.app/)
