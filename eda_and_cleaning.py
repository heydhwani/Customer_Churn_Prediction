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

print("\nAfter basic cleaning:")
print(df.head())

# SAVE CLEANED DATA
clean_path = Path("Data/clean_churn.csv")
df.to_csv(clean_path, index=False)

print("\n✓ Cleaned dataset saved at Data/clean_churn.csv")
