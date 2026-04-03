## Student Performance Intelligence System (SPIS)
A multi-stage machine learning pipeline that predicts student performance, detects at-risk students, and provides actionable recommendations - all in one connected system deployed as a Streamlit dashboard.

## What it does
Most student analytics tools work in isolation. SPIS connects five modules into a single pipeline:
  i. Classification — groups students as good performer or at-risk using logistic regression
  ii. Score Prediction — predicts exact exam scores using linear regression with SHAP explanations
  iii. Risk Detection — flags at-risk students using a random forest classifier
  iv. Counterfactual Guidance — tells at-risk students what to change (attendance, study hours, sleep) and by how much
  v. Resilience Analysis — identifies how low-resource students succeed despite socioeconomic disadvantage using a stacking ensemble

## Dataset
Student Performance Factors dataset by Ahmed et al. (~6,000 records) from Kaggle. Features include attendance, hours studied, tutoring sessions, parental involvement, family income, and motivation level.
Dataset link: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors

## Results
The models outperformed the base paper across all metrics.
Linear Regression: R² 0.726, MAE 0.848, RMSE 1.967
Random Forest (risk detection): 85.5% accuracy, 0.84 recall on at-risk class
Stacking Ensemble: R² 0.727, MAE 0.848, RMSE 1.966

## Key finding: attendance and study hours are the strongest predictors of performance. Resilient students (low income, high performers) compensate entirely through these two factors.

## Run locally
 streamlit run src/app.py

## Stack
Python, Scikit-learn, SHAP, Streamlit, Pandas, Matplotlib
