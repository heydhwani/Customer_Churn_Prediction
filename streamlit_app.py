import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("Bank Customer Churn Prediction App")
st.write("Fill the customer details and click **Predict Churn** to see the result.")


# Dropdown maps 
COUNTRY_MAP = {
    "France 🇫🇷": "France",
    "Spain 🇪🇸": "Spain",
    "Germany 🇩🇪": "Germany"
}

GENDER_MAP = {
    "Male": "Male",
    "Female": "Female"
}

YESNO_MAP = {
    "Yes": 1,
    "No": 0
}


# Streamlit Form UI 
with st.form("churn_form"):
    st.subheader("Customer Information")

    customer_id = st.number_input("Customer ID", value=15634602, help="Unique customer identifier")

    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650,
                                   help="Higher credit score → lower churn risk")

    country_label = st.selectbox("Country", list(COUNTRY_MAP.keys()))
    gender_label = st.selectbox("Gender", list(GENDER_MAP.keys()))

    age = st.number_input("Age", min_value=18, max_value=100, value=42)

    st.subheader("Account Information")

    tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=10, value=3)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    products_number = st.selectbox("Number of Bank Products", [1, 2, 3, 4], index=1)
    credit_card_label = st.selectbox("Has Credit Card?", list(YESNO_MAP.keys()))
    active_member_label = st.selectbox("Active Member?", list(YESNO_MAP.keys()))
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0)

    submitted = st.form_submit_button("Predict Churn")


# Handle form submission
if submitted:
    country = COUNTRY_MAP[country_label]
    gender = GENDER_MAP[gender_label]
    credit_card = YESNO_MAP[credit_card_label]
    active_member = YESNO_MAP[active_member_label]

    # Build payload for FastAPI
    payload = {
        "customer": {
            "customer_id": int(customer_id),
            "credit_score": int(credit_score),
            "country": country,
            "gender": gender,
            "age": int(age)
        },
        "account": {
            "tenure": int(tenure),
            "balance": float(balance),
            "products_number": int(products_number),
            "credit_card": int(credit_card),
            "active_member": int(active_member),
            "estimated_salary": float(estimated_salary)
        }
    }

    st.info("Sending request to API...")

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
    else:
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction received!")

            st.metric("Churn Probability", f"{result['churn_probability']*100:.2f}%")
            st.metric("Churn Prediction", "Yes" if result["prediction"] == 1 else "No")

            st.write("**Remark:**", result["remark"])

            st.divider()
            st.json(result)

        else:
            st.error(f"API error (status {response.status_code})")
            try:
                st.json(response.json())
            except:
                st.text(response.text)
