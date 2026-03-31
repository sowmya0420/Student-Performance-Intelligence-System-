import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the data your friend provided
df = pd.read_csv("Dataset/StudentPerformanceFactors - 6k.csv")

# 2. Create the "At-Risk" label (If score is below median, they are at risk)
threshold = df["Exam_Score"].median()
df["At_Risk"] = (df["Exam_Score"] < threshold).astype(int)

# 3. Pick the features (The data we use to predict)
features = [
"Attendance",
"Hours_Studied",
"Sleep_Hours",
"Previous_Scores",
"Tutoring_Sessions"
]
X = df[features]
y = df["At_Risk"]

# 4. Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Create the Logistic Regression Model
model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_scaled)
print("\nAccuracy:", round(accuracy_score(y_test, y_pred),2))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 7. Save the results into "pkl" files so your friend can use them
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "logistic_scaler.pkl")
joblib.dump(features, "logistic_features.pkl")
joblib.dump(X_train_scaled, "logistic_background.pkl")

print("Finished! You now have the model files ready.")