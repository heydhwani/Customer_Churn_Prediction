import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# LOAD CLEANED DATA

df = pd.read_csv("Data/clean_churn.csv")

print("✓ Cleaned dataset loaded")
print(df.head())

# FEATURES AND TARGET

X = df.drop("churn", axis=1)
y = df["churn"]

# IDENTIFY COLUMN TYPES

# Categorical columns
categorical_cols = ["country", "gender"]

# Numeric columns 
numeric_cols = [
    "credit_score", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member",
    "estimated_salary"
]

print("\nCategorical Columns:", categorical_cols)
print("Numeric Columns:", numeric_cols)

# PREPROCESSING PIPELINE

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

# MODEL

model = RandomForestClassifier(
    n_estimators=200,
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

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# TRAIN MODEL

print("\nTraining model...")
pipeline.fit(X_train, y_train)
print("✓ Model training complete!")

# EVALUATION

y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))