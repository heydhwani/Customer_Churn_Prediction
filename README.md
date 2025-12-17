# 📚  Customer Churn Prediction

A machine-learning-powered web API using FastAPI to predict whether a bank customer is likely to churn (leave the bank) based on demographic and account-related features. This project is built on the popular **Bank Customer Churn Dataset** from Kaggle.

---

## 🚀 Features

- Predicts whether a customer will churn or stay
- Provides churn probability (%).
- Clean input grouping:
  - Customer Information
  - Account Information
- Powerful model trained on real-world banking data.
- Interactive Streamlit web app with real-time predictions
- Fully functional FastAPI backend (local + deployed on Render).
- Easy JSON-based prediction endpoint.
- Professional UI with dropdown mappings to avoid wrong inputs.
- EDA with correlation heatmap, histograms, duplicates handling, and     missing value treatment.
- Gradient Boosting model for high accuracy
- Test App here-
- [https://customerchurnprediction-28uyemartqkvyuqinrm5n5.streamlit.app/](https://customerchurnprediction-28uyemartqkvyuqinrm5n5.streamlit.app/)



---

## 📂 Project Structure

```
Customer_Churn_Prediction/
├── app.py                           # FastAPI backend logic
├
├── churn_simple.py                  # Model training script (Gradient Boosting)
├── eda_and_cleaning.py              # Exploratory Data Analysis + cleaning
├── streamlit_app.py                 # Streamlit frontend for churn prediction
│
├── Data/
│   ├── Bank Customer Churn Prediction.csv
│   ├── clean_churn.csv
│
├── models/
│   ├── churn.joblib                 # Final Gradient Boosting model
│
├── plots/
│   ├── correlation_heatmap.png
│   ├── numeric_histograms.png
│
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation

```

---

## 🛠️ Installation

### Prerequisites:
- Python 3.14.0

### Setup:
```bash
# Repository Name
Customer_Churn_Prediction

# Create virtual environment
python -m venv venv
venv\Scripts\activate 

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Model Training

To train the churn prediction model:

```bash
python churn_simple.py
```

This will generate the final model:
- `models/churn.joblib`

---

## 🚦 Running the API

Start the FastAPI server:
```bash
uvicorn app:app --reload --port 8000

```

Navigate to:
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Root: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Render: [https://customer-churn-prediction-2-jqx5.onrender.com](https://customer-churn-prediction-2-jqx5.onrender.com)

---

## 📥 API Usage

### Endpoint:
```
POST /predict
```

### Request Body Example:
```json
{
  "customer": {
    "customer_id": 15634602,
    "credit_score": 650,
    "country": "France",
    "gender": "Female",
    "age": 42
  },
  "account": {
    "tenure": 3,
    "balance": 50000.0,
    "products_number": 2,
    "credit_card": 1,
    "active_member": 1,
    "estimated_salary": 100000.0
  }
}

```

### Sample Response:
```json
{
  "churn_prediction": "No",
  "probability": 0.18
}

```

---

## 🧠 Model Overview

- Algorithm: Gradient Boosting Classifier
- Accuracy: ~87%
- Goal: Improve recall for class “1” (churn) using class weights
- Input Features: 11
- Target: churn

---

## 🚀 EDA Script Overview
The script performs the following tasks:

### 2. Histograms
Purpose:
- `Shows distribution of age, balance, credit score, tenure, etc`

---

### 4. Correlation Heatmap
Purpose:
- `Used numeric features (customer_id removed).`
- `Helps understand which features influence churn most.`
---

# 🎨Streamlit App

This is a **simple Streamlit web application** that allows users to input customer details and get a predicted final output using a Machine Learning model hosted on a **FastAPI backend**.

The Streamlit app collects user inputs through a clean UI, sends them as JSON to the FastAPI `/predict` endpoint, and displays the predicted output.

---

## 🚀 Features

- Accepts all customer fields
- Sends POST request to FastAPI backend
- Displays predicted churn + probability
- Shows error messages for invalid inputs

---

## 📦 How to Run the Streamlit App

### 1️⃣ Install required libraries
```bash
pip install streamlit requests
```
### 2️⃣ Run the app
```bash
streamlit run streamlit_app.py
```

---

## DATASET LINK:
https://www.kaggle.com/datasets/marslinoedward/bank-customer-churn-prediction


---



