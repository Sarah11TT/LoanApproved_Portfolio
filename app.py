import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import numpy as np
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Loan Risk AI Portfolio",
    page_icon="🏦",
    layout="wide"
)

# --- 2. DATA & MODEL LOADING ---
@st.cache_data
def load_essentials():
    # Load the processed data
    df = pd.read_csv('data/processed/cleaned_loan_data.csv')
    
    # Load the model and the column structure from training
    with open('models/best_loan_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
        
    return df, model, model_columns

# Try to load, or show error if user hasn't run the scripts yet
try:
    df, best_model, model_columns = load_essentials()
except FileNotFoundError:
    st.error("Missing files! Please run 'src/data_cleaning.py' and 'src/model_training.py' first.")
    st.stop()

# --- 3. SIDEBAR / NAVIGATION ---
st.sidebar.title("Loan Portfolio App")
# We use a radio button for navigation, but you can also use tabs
page = st.sidebar.radio("Navigate to:", ["Executive Dashboard", "AI Risk Predictor"])

# --- PAGE 1: EXECUTIVE DASHBOARD (From your first App.py) ---
if page == "Executive Dashboard":
    st.title("📊 Loan Approval Executive Dashboard")
    st.markdown("Analyzing key risk drivers for the current loan portfolio.")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Apps", f"{len(df):,}")
    c2.metric("Avg Credit Score", int(df['CreditScore'].mean()))
    c3.metric("Approval Rate", f"{(df['LoanApproved'].mean()*100):.1f}%")
    c4.metric("Avg Loan Amount", f"${df['LoanAmount'].mean():,.0f}")

    st.divider()

    # Charts (This is the logic from your first app.py)
    row1_1, row1_2 = st.columns(2)
    with row1_1:
        fig1 = px.histogram(df, x="CreditScore", color="LoanApproved", 
                            title="Credit Score Distribution by Approval", 
                            barmode="group", color_discrete_map={1: 'green', 0: 'red'})
        st.plotly_chart(fig1, use_container_width=True)
        
    with row1_2:
        # Sunburst showing how Education and Employment impact approval
        fig2 = px.sunburst(df, path=['Education', 'EmploymentType'], values='LoanApproved',
                           title="Approval Clusters by Background")
        st.plotly_chart(fig2, use_container_width=True)

# --- PAGE 2: AI RISK PREDICTOR (From your second App.py) ---
else:
    st.title("🔮 AI Loan Approval Predictor")
    st.write("This tool uses the **best-performing model** from our training pipeline (Random Forest/Gradient Boosting).")

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 30)
            income = st.number_input("Annual Income ($)", 0, 200000, 50000)
            loan = st.number_input("Loan Amount Requested ($)", 0, 100000, 15000)
            exp = st.number_input("Years Experience", 0, 50, 5)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            edu = st.selectbox("Education", sorted(df['Education'].unique()))
            emp = st.selectbox("Employment Type", sorted(df['EmploymentType'].unique()))
            city = st.selectbox("City", sorted(df['City'].unique()))
            score = st.slider("Credit Score", 300, 850, 650)
        
        submit = st.form_submit_button("Run Risk Assessment")

    if submit:
        # 1. Create a dictionary of the input (Exactly as in your second code)
        input_dict = {
            'Age': age, 'Income': income, 'LoanAmount': loan, 
            'CreditScore': score, 'YearsExperience': exp,
            'Gender': gender, 'Education': edu, 
            'EmploymentType': emp, 'City': city,
            'DTI_Ratio': loan / (income + 1)
        }
        
        # 2. Convert to DataFrame and apply Dummies (The encoding logic you requested)
        input_df = pd.DataFrame([input_dict])
        input_df_encoded = pd.get_dummies(input_df)

        # 3. Ensure columns match the training set exactly (The alignment logic)
        final_input = pd.DataFrame(columns=model_columns)
        for col in model_columns:
            if col in input_df_encoded.columns:
                final_input[col] = input_df_encoded[col]
            else:
                final_input[col] = 0
        
        # Re-order columns to match model expectations
        final_input = final_input[model_columns].fillna(0)
        
        # 4. Prediction
        prob = best_model.predict_proba(final_input)[0][1]
        
        st.divider()
        if prob > 0.5:
            st.success(f"### ✅ Likely Approved (Probability: {prob:.1%})")
            st.balloons()
        else:
            st.error(f"### ❌ Likely Denied (Probability: {prob:.1%})")
            st.warning("Recommendation: Review debt-to-income levels or improve Credit Score.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed by a Master of Data Science Graduate.")