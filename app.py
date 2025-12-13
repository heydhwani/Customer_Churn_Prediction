from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Load Model 
try:
    model = joblib.load("models/churn_rf_balanced.joblib")
except:
    raise RuntimeError("Model not found! Ensure churn_rf_balanced.joblib exists in models/")


class CustomerInfo(BaseModel):
    customer_id: int = Field(..., example=15634602)
    credit_score: int = Field(..., ge=300, le=900, example=650)
    country: str = Field(..., example="France")
    gender: str = Field(..., example="Female")
    age: int = Field(..., ge=18, le=100, example=42)


# Account Details

class AccountInfo(BaseModel):
    tenure: int = Field(..., ge=0, le=10, example=3)
    balance: float = Field(..., example=50000.0)
    products_number: int = Field(..., ge=1, le=4, example=2)
    credit_card: int = Field(..., ge=0, le=1, example=1)
    active_member: int = Field(..., ge=0, le=1, example=1)
    estimated_salary: float = Field(..., example=100000.0)


# Combined Input Model

class ChurnInput(BaseModel):
    customer: CustomerInfo
    account: AccountInfo

# Initialize FastAPI App
app = FastAPI(title="Customer Churn Prediction API")

@app.get("/")
def root():
    return {"message": "Welcome to the Customer Churn Prediction API!"}


# Prediction Endpoint

@app.post("/predict")
def predict_churn(data: ChurnInput):
    try:
        # Flatten data sections into a single dict
        input_data = {
            **data.customer.dict(),
            **data.account.dict()
        }

        # Convert to DataFrame in model's expected order
        df = pd.DataFrame([input_data])

        # If feature order matters
        try:
            df = df[model.feature_names_in_]
        except:
            pass  

        # Predict probability & class
        prob = model.predict_proba(df)[0][1]
        pred = int(model.predict(df)[0])

        
        remark = "Customer likely to churn" if pred == 1 else "Customer likely to stay"

        return {
            "customer_id": data.customer.customer_id,
            "churn_probability": round(float(prob), 4),
            "prediction": pred,
            "remark": remark
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
