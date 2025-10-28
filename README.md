# Customer-Churn-Prediction
Customer Churn Prediction — Telecom Industry  
Predicting which customers are likely to leave a telecom company using machine learning

# Project Overview
Customer churn is one of the most pressing challenges in the telecom industry. This project builds a machine learning model that predicts whether a customer will churn (leave) or stay based on their account information, usage behavior, and subscription type.
Using Python and real-world telecom data, the project demonstrates an end-to-end data science workflow — from data cleaning and feature engineering to model evaluation and interpretation.

# Key Objectives
. Identify the key factors influencing customer churn.
. Build a predictive model that classifies customers as "likely to churn" or "likely to stay."
. Provide actionable insights for customer retention strategies.
. Create a foundation for deploying the model via a Streamlit dashboard (optional extension).

| Category                    | Tools                                                             |
| --------------------------- | ----------------------------------------------------------------- |
| Language                    | Python                                                            |
| Libraries                   | pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, joblib |
| Visualization               | matplotlib, seaborn                                               |
| Model Deployment (optional) | Streamlit                                                         |
| Environment                 | Google Colab / Jupyter Notebook                                   |


# Project Workflow
1. Data Loading & Cleaning
  .  Removed irrelevant columns (e.g., customerID)
  .  Converted data types and handled missing values
  .  Encoded categorical variables

3. Exploratory Data Analysis (EDA)
  . Visualized churn distribution and feature relationships
  . Identified trends in tenure, contract type, and payment method

4. Feature Engineering & Scaling
  . Created dummy variables
  . Scaled numeric features (tenure, MonthlyCharges, TotalCharges)

5. Modeling
  . Trained multiple models:
    . Logistic Regression
    . Random Forest Classifier
    . XGBoost Classifier
  . Compared performance based on Accuracy, ROC-AUC, and Confusion Matrix

6. Model Evaluation & Insights
  . Assessed feature importance
  . Visualized ROC curves
  . Interpreted model outputs

7. Model Saving
  . Saved trained model using joblib (churn_model.pkl)
  . Prepared for deployment in a web app or API

# Results
  . Best Model: Random Forest Classifier
  . Accuracy: ~80–85% (depending on parameter tuning)
  . Key Drivers of Churn:
    . Contract Type (month-to-month customers churn more)
    . Tenure (newer customers are more likely to churn)
    . Internet Service & Monthly Charges
