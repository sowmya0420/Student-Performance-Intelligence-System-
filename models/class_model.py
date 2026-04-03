# =========================
# model_training.py
# =========================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================
# Load Dataset
# =========================
df = pd.read_csv("Dataset/StudentPerformanceFactors - 6k.csv")

# =========================
# Create Balanced At-Risk Label (Using Median)
# =========================
threshold = df["Exam_Score"].median()
print("Using threshold:", threshold)

df["At_Risk"] = (df["Exam_Score"] < threshold).astype(int)

print("Class Distribution:")
print(df["At_Risk"].value_counts())

# =========================
# Select Features
# =========================
features = [
    "Attendance",
    "Hours_Studied",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions"
]

X = df[features]
y = df["At_Risk"]

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Train Model
# =========================
model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# =========================
# Evaluation
# =========================
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.savefig("confusion_matrix.png")

# =========================
# Save Files
# =========================
joblib.dump(model, "clf_model.pkl")
joblib.dump(scaler, "clf_scaler.pkl")
joblib.dump(features, "clf_features.pkl")

print("\nModel training completed successfully.")
