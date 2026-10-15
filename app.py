import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🧑‍💼",
    layout="wide"
)

# Custom CSS styling for a gorgeous look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
    }
    .stButton>button:hover {
        background-color: #ff2222;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('employee_model.pkl')

model = load_model()

# Header Section
st.title("🧑‍💼 Employee Attrition Prediction System")
st.markdown("Use this intelligent HR tool to predict whether an employee is at risk of leaving the company based on key workplace and personal factors.")
st.markdown("---")

# Input Form Layout using Columns
st.subheader("📝 Enter Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income (₹)", min_value=1000, max_value=500000, value=50000, step=1000)
    distance_from_home = st.slider("Distance From Home (KM)", 1, 30, 5)

with col2:
    over_time = st.selectbox("Over Time", ["Yes", "No"])
    job_satisfaction = st.selectbox("Job Satisfaction Level", [1, 2, 3, 4], format_func=lambda x: f"Level {x}")
    environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], format_func=lambda x: f"Level {x}")

with col3:
    total_working_years = st.slider("Total Working Years", 0, 40, 5)
    years_at_company = st.slider("Years At Company", 0, 40, 3)
    job_role = st.selectbox("Job Role", [
        'Sales Executive', 'Research Scientist', 'Laboratory Technician', 
        'Manufacturing Director', 'Healthcare Representative', 'Manager', 
        'Sales Representative', 'Research Director', 'Human Resources'
    ])

st.markdown("---")

# Prediction logic when button is clicked
if st.button("Predict Attrition Risk"):
    # Create a base dataframe filled with default/average values matching your training columns
    input_data = pd.DataFrame(0, index=[0], columns=model.feature_names_in_)
    
    # Map user inputs into the dataframe columns
    if 'Age' in input_data.columns: input_data['Age'] = age
    if 'MonthlyIncome' in input_data.columns: input_data['MonthlyIncome'] = monthly_income
    if 'DistanceFromHome' in input_data.columns: input_data['DistanceFromHome'] = distance_from_home
    if 'OverTime' in input_data.columns: input_data['OverTime'] = 1 if over_time == 'Yes' else 0
    if 'JobSatisfaction' in input_data.columns: input_data['JobSatisfaction'] = job_satisfaction
    if 'EnvironmentSatisfaction' in input_data.columns: input_data['EnvironmentSatisfaction'] = environment_satisfaction
    if 'TotalWorkingYears' in input_data.columns: input_data['TotalWorkingYears'] = total_working_years
    if 'YearsAtCompany' in input_data.columns: input_data['YearsAtCompany'] = years_at_company
    
    # Handle One-Hot Encoded Job Role column dynamically if it exists
    role_column = f"JobRole_{job_role}"
    if role_column in input_data.columns:
        input_data[role_column] = 1

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display Results visually
    st.markdown("### 📊 Prediction Result")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk of Attrition!** This employee is likely to leave the company. (Confidence: {prediction_proba[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ **Low Risk of Attrition.** This employee is likely to stay. (Confidence: {prediction_proba[0][0]*100:.2f}%)")