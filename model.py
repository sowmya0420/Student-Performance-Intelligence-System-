# ============================================================
# STUDENT PERFORMANCE PREDICTION
# Model Training using Selected 8 Features
# ============================================================

# ============================================================
# 1️⃣ IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error


# ============================================================
# 2️⃣ LOAD DATASET
# ============================================================
df = pd.read_csv("Dataset/StudentPerformanceFactors - 6k.csv")

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)


# ============================================================
# 3️⃣ SELECT IMPORTANT FEATURES (8 out of 27)
# ============================================================
selected_features = [
    "Attendance",
    "Hours_Studied",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Access_to_Resources",
    "Parental_Involvement",
    "Family_Income",
    "Motivation_Level"
]

X = df[selected_features]
y = df["Exam_Score"]

print("Selected Feature Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# 4️⃣ IDENTIFY NUMERICAL & CATEGORICAL COLUMNS
# ============================================================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

print("Numerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)


# ============================================================
# 5️⃣ HANDLE MISSING VALUES
# ============================================================
cat_imputer = SimpleImputer(strategy="most_frequent")
X.loc[:, cat_cols] = cat_imputer.fit_transform(X[cat_cols])

print("Remaining Missing Values:", X.isnull().sum().sum())


# ============================================================
# 6️⃣ ENCODE CATEGORICAL FEATURES
# ============================================================
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print("Encoded Feature Shape:", X.shape)


# ============================================================
# 7️⃣ TRAIN-TEST SPLIT & SCALING
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ============================================================
# 8️⃣ BASELINE MODEL – # LINEAR REGRESSION
# ============================================================
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print("\n Baseline Linear Regression Performance")
print("MAE:", round(mean_absolute_error(y_test, y_pred_lr),4))
print("RMSE:", round(rmse_lr, 4))
print("R2:", round(r2_lr, 4))


# ============================================================
# 🔟 STACKING ENSEMBLE MODEL
# ============================================================
estimators = [
    ("ridge", Ridge()),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ("gbr", GradientBoostingRegressor(random_state=42))
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    passthrough=False,
    n_jobs=-1
)

stack.fit(X_train, y_train)
stack_pred = stack.predict(X_test)


rmse_stack = np.sqrt(mean_squared_error(y_test, stack_pred))
r2_stack = r2_score(y_test, stack_pred)

print("\nStacking Ensemble Performance")
print("MAE:", round(mean_absolute_error(y_test, stack_pred),4))
print("RMSE:", round(rmse_stack, 4))
print("R2:", round(r2_stack, 4))
# ============================================================
# SAVE FINAL MODEL (BASELINE)
# ============================================================
joblib.dump(lr, "reg_model.pkl")
joblib.dump(scaler, "reg_scaler.pkl")
joblib.dump(X.columns.tolist(), "reg_columns.pkl")
joblib.dump(X_train, "reg_background.pkl")
joblib.dump(stack, "stack_model.pkl")
print("\n✅ Model and preprocessing objects saved successfully.")