import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier


#  Load cleaned data

df = pd.read_csv("Data/clean_churn.csv")
print("Loaded data shape:", df.shape)
print(df.head())


# Prepare features & target

X = df.drop(columns=["churn", "customer_id"], errors="ignore")
y = df["churn"]


categorical_cols = [c for c in ["country", "gender"] if c in X.columns]
numeric_cols = [c for c in [
    "credit_score", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member", "estimated_salary"
] if c in X.columns]

print("Categorical cols:", categorical_cols)
print("Numeric cols:", numeric_cols)


# Preprocessor


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder="drop"
)


# Build full pipeline

model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


# Train the model

print("\nTraining RandomForest (class_weight='balanced') ...")
pipeline.fit(X_train, y_train)
print("Training complete")


# Evaluate

y_pred = pipeline.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Save pipeline

Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/churn_rf_balanced.joblib")
print("\nSaved pipeline to models/churn_rf_balanced.joblib")
