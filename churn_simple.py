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