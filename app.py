import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("EMI Eligibility Prediction App")

clf_model = joblib.load("clf_model.pkl")
reg_model = joblib.load("reg_model.pkl")


# User input form
age = st.number_input("Age", min_value=18, max_value=100, value=30)
education = st.selectbox("Education Level (0=none,1=High School,2=Graduate,3=Postgraduate)", [0,1,2,3], index=2)
monthly_salary = st.number_input("Monthly Salary", min_value=0, value=50000)
years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
monthly_rent = st.number_input("Monthly Rent", min_value=0, value=10000)
family_size = st.number_input("Family Size", min_value=1, value=3)
dependents = st.number_input("Number of Dependents", min_value=0, value=2)
school_fees = st.number_input("School Fees", min_value=0, value=0)
college_fees = st.number_input("College Fees", min_value=0, value=0)
travel_expenses = st.number_input("Travel Expenses", min_value=0, value=2000)
groceries_utilities = st.number_input("Groceries / Utilities", min_value=0, value=5000)
other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0, value=1000)
current_emi_amount = st.number_input("Current EMI Amount", min_value=0, value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
bank_balance = st.number_input("Bank Balance", min_value=0, value=10000)
emergency_fund = st.number_input("Emergency Fund", min_value=0, value=5000)
requested_amount = st.number_input("Requested Loan Amount", min_value=0, value=50000)
requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, value=12)
# max_monthly_emi = st.number_input("Max Affordable EMI", min_value=0, value=5000)
   
gender = st.selectbox("Gender", ["Male","Female"])
marital_status = st.selectbox("Marital Status", ["Single","Married"])
employment_type = st.selectbox("Employment Type", ["Private","Self-employed"])
company_type = st.selectbox("Company Type", ["MNC","Mid-size","Small","Startup"])
house_type = st.selectbox("House Type", ["Own","Rented"])
existing_loans = st.selectbox("Existing Loans?", ["No","Yes"])
emi_scenario = st.selectbox("EMI Scenario", ["Education EMI","Home Appliances EMI","Personal Loan EMI","Vehicle EMI"])
    
# Convert manually (same as training!)
edu_map = {'High School':0, 'Graduate':1, 'Post Graduate':2, 'Professional':3}
eleg_map={2:"Not_Eligible",0:"Eligible",1:"High_Risk"}


# 'age', 'education', 'monthly_salary', 'years_of_employment',
#        'monthly_rent', 'family_size', 'dependents', 'school_fees',
#        'college_fees', 'travel_expenses', 'groceries_utilities',
#        'other_monthly_expenses', 'current_emi_amount', 'credit_score',
#        'bank_balance', 'emergency_fund', 'requested_amount',
#        'requested_tenure', 'gender_M', 'marital_status_Single',
#        'employment_type_Private', 'employment_type_Self-employed',
#        'company_type_MNC', 'company_type_Mid-size', 'company_type_Small',
#        'company_type_Startup', 'house_type_Own', 'house_type_Rented',
#        'existing_loans_Yes', 'emi_scenario_Education EMI',
#        'emi_scenario_Home Appliances EMI', 'emi_scenario_Personal Loan EMI',
#        'emi_scenario_Vehicle EMI']


input_dict = {
        "age": age,
        "education": education,
        "monthly_salary": monthly_salary,
        "years_of_employment": years_of_employment,
        "monthly_rent": monthly_rent,
        "family_size": family_size,
        "dependents": dependents,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        # "max_monthly_emi": max_monthly_emi,
        # "gender_Female": 1 if gender=="Female" else 0,
        "gender_M": 1 if gender=="Male" else 0,
        "marital_status_Single": 1 if marital_status=="Single" else 0,
        # "marital_status_Married": 1 if marital_status=="Married" else 0,
        "employment_type_Private": 1 if employment_type=="Private" else 0,
        "employment_type_Self-employed": 1 if employment_type=="Self-employed" else 0,
        "company_type_MNC": 1 if company_type=="MNC" else 0,
        "company_type_Mid-size": 1 if company_type=="Mid-size" else 0,
        "company_type_Small": 1 if company_type=="Small" else 0,
        "company_type_Startup": 1 if company_type=="Startup" else 0,
        "house_type_Own": 1 if house_type=="Own" else 0,
        "house_type_Rented": 1 if house_type=="Rented" else 0,
        "existing_loans_Yes": 1 if existing_loans=="Yes" else 0,
        "emi_scenario_Education EMI": 1 if emi_scenario=="Education EMI" else 0,
        "emi_scenario_Home Appliances EMI": 1 if emi_scenario=="Home Appliances EMI" else 0,
        "emi_scenario_Personal Loan EMI": 1 if emi_scenario=="Personal Loan EMI" else 0,
        "emi_scenario_Vehicle EMI": 1 if emi_scenario=="Vehicle EMI" else 0
    }


# Convert to DataFrame
df = pd.DataFrame([input_dict])

if st.button("Check Eligibility"):
    pred_class = clf_model.predict(df)[0]
    pred_emi = reg_model.predict(df)[0]

    st.subheader("Prediction Result:")
    st.write("**Eligibility:**", eleg_map[pred_class])
    if pred_class==1 or pred_class==2:
    #   st.write("**Max EMI Allowed:** ₹", 0)
        pass
    else:
      st.write("**Max EMI Allowed:** ₹", int(pred_emi))







