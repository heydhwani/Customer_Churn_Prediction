import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("Data/Bank Customer Churn Prediction.csv")
print("Rows, cols:", df.shape)
# Quick look
print(df.head().T)

#Basic cleaning
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])