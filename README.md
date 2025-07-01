# Customer Churn Analysis and Prediction

This project focuses on analyzing customer churn data and building a classification model to predict whether a customer will churn or not. It includes detailed exploratory data analysis (EDA), preprocessing, model training, and deployment using a Streamlit web application.

---

## Project Overview

- Performed univariate analysis to understand individual feature distributions.
- Conducted bivariate analysis to explore relationships between features and the target variable.
- Applied multivariate analysis to assess interactions among multiple variables.
- Removed unnecessary columns such as `CustomerID`, and highly correlated features like `TotalCharges`.
- Addressed class imbalance using SMOTE and class weighting techniques.

---

## Models Used

Several machine learning models were trained and evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost (Best Performing Model)

---

## Preprocessing Steps

- One-hot encoding was applied to categorical features like `ContractType` and `InternetService`.
- Label encoding was used for binary features such as `Gender` and `TechSupport`.
- Log transformation was applied to skewed numerical features such as `Tenure` and `TotalCharges`.
- Feature scaling was performed using StandardScaler for certain models.
- Boolean-like columns were converted from `True`/`False` to `1`/`0`.

---

## Best Performing Model

The XGBoost Classifier showed the highest accuracy, F1-score, and recall across all models and was chosen as the final model for deployment. It was saved as `XGBOOST.pkl`.

---

## Deployment with Streamlit

The trained model was integrated into a user-friendly Streamlit web application, which takes user input and predicts customer churn in real time.

### Required `.pkl` Files for App:

- `XGBOOST.pkl` – Final trained model
- `scaler.pkl` – Scaler used during preprocessing
- `techsupport_encoder.pkl` – Label encoder for the `TechSupport` feature
- `expected_columns.pkl` – Column ordering used for model input



