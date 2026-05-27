import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Load dataset
df = pd.read_csv("loan_approval_data.csv")

# Drop Applicant ID if present
if "Applicant_ID" in df.columns:
    df = df.drop("Applicant_ID", axis=1)

df = df.fillna(df.mode().iloc[0])

# Encode categorical columns
label_encoders = {}

for column in df.columns:
    if df[column].dtype == object:
        df[column] = df[column].astype(str)

        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])

        label_encoders[column] = le

# Features and target
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

# Train model
model = GaussianNB()
model.fit(X, y)

# Streamlit UI
st.title("AI-Based Loan Prediction System")

st.write("Enter applicant details below:")

# User Inputs
Applicant_Income = st.number_input("Applicant Income", min_value=0)

Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0)

Age = st.number_input("Age", min_value=18, max_value=100)

Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900)

Loan_Amount = st.number_input("Loan Amount", min_value=0)

Loan_Term = st.number_input("Loan Term (Months)", min_value=1)

Dependents = st.number_input("Dependents", min_value=0)

Existing_Loans = st.number_input("Existing Loans", min_value=0)

DTI_Ratio = st.number_input("DTI Ratio", min_value=0.0)

Savings = st.number_input("Savings", min_value=0)

Collateral_Value = st.number_input("Collateral Value", min_value=0)

Employment_Status = st.selectbox(
    "Employment Status",
    label_encoders["Employment_Status"].classes_
)

Marital_Status = st.selectbox(
    "Marital Status",
    label_encoders["Marital_Status"].classes_
)

Loan_Purpose = st.selectbox(
    "Loan Purpose",
    label_encoders["Loan_Purpose"].classes_
)

Property_Area = st.selectbox(
    "Property Area",
    label_encoders["Property_Area"].classes_
)

Education_Level = st.selectbox(
    "Education Level",
    label_encoders["Education_Level"].classes_
)

Gender = st.selectbox(
    "Gender",
    label_encoders["Gender"].classes_
)

Employer_Category = st.selectbox(
    "Employer Category",
    label_encoders["Employer_Category"].classes_
)

# Encode inputs
Employment_Status = label_encoders["Employment_Status"].transform([Employment_Status])[0]
Marital_Status = label_encoders["Marital_Status"].transform([Marital_Status])[0]
Loan_Purpose = label_encoders["Loan_Purpose"].transform([Loan_Purpose])[0]
Property_Area = label_encoders["Property_Area"].transform([Property_Area])[0]
Education_Level = label_encoders["Education_Level"].transform([Education_Level])[0]
Gender = label_encoders["Gender"].transform([Gender])[0]
Employer_Category = label_encoders["Employer_Category"].transform([Employer_Category])[0]

# Prediction
if st.button("Predict Loan Status"):

    input_data = pd.DataFrame([[
        Applicant_Income,
        Coapplicant_Income,
        Employment_Status,
        Age,
        Marital_Status,
        Dependents,
        Credit_Score,
        Existing_Loans,
        DTI_Ratio,
        Savings,
        Collateral_Value,
        Loan_Amount,
        Loan_Term,
        Loan_Purpose,
        Property_Area,
        Education_Level,
        Gender,
        Employer_Category
    ]], columns=X.columns)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
