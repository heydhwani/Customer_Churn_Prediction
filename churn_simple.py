
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier


# LOAD CLEANED DATA

df = pd.read_csv("Data/clean_churn.csv")

print("✓ Cleaned dataset loaded")
print(df.head())

# FEATURES AND TARGET

X = df.drop("churn", axis=1)
y = df["churn"]

categorical_cols = ["country", "gender"]

numeric_cols = [
    "credit_score", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member",
    "estimated_salary"
]


# PREPROCESSING

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)


# XGBOOST MODEL (High Accuracy)

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])


# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# TRAIN MODEL

print("\nTraining XGBoost model...")
pipeline.fit(X_train, y_train)
print("✓ Training complete!")


# EVALUATION

y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# SAVE MODEL

Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/churn.joblib")

print("\n✓ High-accuracy model saved at models/churn.joblib")
