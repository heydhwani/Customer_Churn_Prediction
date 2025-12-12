import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# LOAD DATA

data_path = Path("Data/Bank Customer Churn Prediction.csv")
df = pd.read_csv(data_path)

print("\n✓ Raw dataset loaded")
print(df.head())

# BASIC CLEANING

# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(method='ffill')

# Drop columns that do not help in prediction
cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

