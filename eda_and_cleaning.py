import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

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

# SIMPLE EDA
print("\nShape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

print("\nChurn distribution:")
print(df["churn"].value_counts())

# Correlation (NUMERIC ONLY BUT REMOVE customer_id)


num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove customer_id because it does not represent a numeric feature
if "customer_id" in num_cols:
    num_cols.remove("customer_id")

corr = df[num_cols].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (customer_id removed)")
plt.savefig(plots_dir/ "correlation_heatmap.png", dpi=300, bbox_inches="tight")

plt.show()

# HISTOGRAMS FOR NUMERICAL COLUMNS


num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove customer_id (we don't want to visualize ID values)
if "customer_id" in num_cols:
    num_cols.remove("customer_id")

print("\nPlotting Histograms for Numeric Features...")

df[num_cols].hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.tight_layout()

plt.savefig(plots_dir/"numeric_histograms.png", dpi=300, bbox_inches="tight")

plt.show()

print("\n✓ Histograms displayed successfully!")

