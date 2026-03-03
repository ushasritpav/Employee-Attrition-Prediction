import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load feature order
with open("features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.set_page_config(page_title="Employee Attrition Predictor")

st.title("Employee Attrition Prediction")
st.write("Enter employee details to predict the likelihood of attrition.")

# ---------------- USER INPUTS ---------------- #

age = st.slider("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
overtime = st.selectbox("OverTime", ["No", "Yes"])
daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800)
monthly_rate = st.number_input("Monthly Rate", min_value=1000, max_value=30000, value=15000)

# Convert categorical input
overtime_value = 1 if overtime == "Yes" else 0

# Create empty dataframe with correct feature columns
# Load original dataset to compute baseline values
data = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Convert Attrition column to numeric (like training)
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

# Drop unused columns (same as training)
data = data.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

# Encode categorical columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

# Create baseline input using mean values
input_data = pd.DataFrame([data.mean()], columns=data.columns)

# Replace selected fields with user input
input_data["Age"] = age
input_data["MonthlyIncome"] = monthly_income
input_data["OverTime"] = overtime_value
input_data["DailyRate"] = daily_rate
input_data["MonthlyRate"] = monthly_rate

# Ensure correct feature order
input_data = input_data[feature_columns]

# Scale input
input_scaled = scaler.transform(input_data)

# Predict button
if st.button("Predict Attrition"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Attrition 🚨 (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk of Attrition ✅ (Probability: {probability:.2f})")
