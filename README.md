# 🏠 House Price Prediction System

An end-to-end machine learning application that predicts house sale prices based on real property features, built as part of an AI/ML Internship (Task 4 — End-to-End ML Application with Deployment).

**🔗 Live App:** [house-price-prediction-task4.streamlit.app](https://house-price-prediction-task4.streamlit.app)

---

## Overview

This project takes real-world housing data, processes it, trains and compares multiple machine learning models, tunes the best-performing one, and deploys it as an interactive Streamlit web application where users can input house details and receive a predicted sale price along with supporting insights.

---

## Dataset

- **Source:** Ames Housing Dataset (House Prices – Advanced Regression Techniques)
- **Size:** 1,460 houses, 80 original features
- **Target variable:** `SalePrice`

---

## Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Modeling:** Scikit-learn (Linear Regression, Ridge Regression, Random Forest, Gradient Boosting)
- **Visualization:** Matplotlib
- **Web App:** Streamlit
- **Model Persistence:** Joblib
- **Version Control / Deployment:** Git, GitHub, Streamlit Community Cloud

---

## Project Workflow

1. **Data Processing** — handled missing values (context-aware, not blind imputation), removed duplicates, detected and removed outliers, encoded categorical variables (ordinal + one-hot), engineered new features (`TotalSF`, `HouseAge`, `TotalBath`, etc.), and scaled continuous numeric features.
2. **Model Development** — trained and compared 4 models: Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting.
3. **Hyperparameter Tuning** — used `RandomizedSearchCV` (30 combinations, 5-fold cross-validation) to tune the best-performing model.
4. **Model Saving** — final tuned model and preprocessing objects (scaler, feature columns, defaults) saved with `joblib`.
5. **Streamlit Application** — interactive multi-tab app for prediction, data insights, and model performance comparison.
6. **Deployment** — deployed live via Streamlit Community Cloud, connected to this GitHub repository.

---

## Model Performance

| Model                         | MAE        | RMSE       | R² Score   |
|-------------------------------|------------|------------|------------|
| **Gradient Boosting (Tuned)** | **13,887** | **19,966** | **0.9273** |
| Random Forest                 | 16,134     | 23,910     | 0.8965     |
| Ridge Regression              | 18,812     | 25,530     | 0.8820     |
| Linear Regression             | 19,588     | 27,460     | 0.8635     |

**Best model:** Gradient Boosting Regressor, tuned with `RandomizedSearchCV`, explaining ~92.7% of the variance in house prices.

---

## App Features

- Interactive input form for house details (size, rooms, quality, location, age)
- Prediction with an error-margin range (± RMSE) instead of a raw single number
- Feature importance visualization (model-wide, adapts to tree-based or linear models)
- Data insights page with price distribution, price vs. living area, price by neighborhood, and feature correlations
- Model performance page comparing all 4 trained models, plus before/after tuning results
- Multiple model selection — choose which trained model makes the prediction
- Light/Dark appearance toggle

---

## Project Structure

```
house-price-prediction-task4/
├── app/
│   ├── app.py                  # Streamlit application
│   └── requirements.txt        # Dependencies (used by Streamlit Cloud deployment)
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── data_description.txt
│   └── processed_train.csv
├── models/
│   ├── final_model.pkl         # Tuned Gradient Boosting model (final)
│   ├── all_models.pkl          # All 4 trained models (for model selection feature)
│   ├── scaler.pkl
│   ├── cols_to_scale.pkl
│   ├── feature_columns.pkl
│   └── feature_defaults.pkl
├── notebooks/
│   └── Task_04_ML.ipynb        # Full data processing & model development notebook
├── report/
│   └── Task4_Report.pdf
├── generate_defaults.py        # Generates default feature values for the app's input form
├── retrain_local.py            # Retrains and saves the final model locally
├── save_all_models.py          # Trains and saves all 4 models for model selection
├── .gitignore
└── README.md
```

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/maryamkhan68/house-price-prediction-task4.git
cd house-price-prediction-task4

# Install dependencies
pip install -r app/requirements.txt

# Run the app
cd app
streamlit run app.py
```

---

## Author

**Maryam Khan**
AI/ML Internship — Task 4
Devixo Solutions