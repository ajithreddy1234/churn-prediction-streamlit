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
##  Model Performance Comparison

| Model            | Accuracy | Precision (0) | Recall (0) | F1-Score (0) | Precision (1) | Recall (1) | F1-Score (1) |
|------------------|----------|---------------|------------|--------------|----------------|------------|--------------|
| Logistic Regression | 0.86     | 0.48          | 0.97       | 0.64         | 0.99           | 0.84       | 0.91         |
| K-Nearest Neighbors| 0.84     | 0.41          | 0.58       | 0.48         | 0.93           | 0.88       | 0.90         |
| Support Vector Machine | 0.88  | 0.54          | 0.58       | 0.56         | 0.93           | 0.93       | 0.93         |
| Decision Tree     | 0.96     | 0.77          | 1.00       | 0.87         | 1.00           | 0.95       | 0.98         |
| Random Forest     | 0.96     | 0.75          | 1.00       | 0.86         | 1.00           | 0.95       | 0.97         |
| XGBoost           | 0.96     | 0.77          | 1.00       | 0.87         | 1.00           | 0.95       | 0.98         |
| Gradient Boosting | 0.94     | 0.70          | 1.00       | 0.82         | 1.00           | 0.94       | 0.97         |

 **Note:**  
All model evaluation metrics, visualizations, and comparison results are thoroughly documented in the Jupyter notebook **`churn_data_cleaning_modeling.ipynb`**. This includes detailed preprocessing steps, model training, performance reports, and rationale for model selection.



