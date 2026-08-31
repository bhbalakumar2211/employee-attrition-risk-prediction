import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="HR Analytics Dashboard", layout="wide", page_icon="🏢")

# 2. Load Data and Models
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_employee_data.csv')

@st.cache_resource
def load_models():
    le = joblib.load('label_encoders.pkl')
    rf_a = joblib.load('attrition_model.pkl')
    rf_p = joblib.load('performance_model.pkl')
    return le, rf_a, rf_p

df = load_data()
label_encoders, attrition_model, performance_model = load_models()

# 3. App Header
st.title("🏢 HR Analytics: Attrition & Performance Predictor")
st.markdown("Use this dashboard to analyze workforce trends and predict employee behavior using Machine Learning.")

# 4. Sidebar Navigation
menu = st.sidebar.radio("Go to:", ["📊 Overview Dashboard", "🤖 Predict Employee Future"])

# ==========================================
# SECTION 1: The Dashboard
# ==========================================
if menu == "📊 Overview Dashboard":
    st.header("Workforce Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees in System", len(df))
    
    attrition_rate = (df['Attrition'].astype(str).str.strip().str.lower() == 'yes').mean() * 100
        
    col2.metric("Overall Attrition Rate", f"{attrition_rate:.1f}%")
    
    avg_income = pd.to_numeric(df['MonthlyIncome'], errors='coerce').mean()
    col3.metric("Average Monthly Income", f"${avg_income:,.0f}")
    
    st.divider()
    
    st.subheader("Attrition Counts by Department")
    st.write("This chart shows the volume of employees leaving from each department.")
    
    # Calculate attrition count by department
    dept_attrition = (
    df[df['Attrition'].astype(str).str.strip().str.lower() == 'yes']
    	.groupby('Department')
    	.size()
    	.sort_values(ascending=True)
    )

    st.bar_chart(dept_attrition)

# ==========================================
# SECTION 2: The Machine Learning Predictor
# ==========================================
elif menu == "🤖 Predict Employee Future":
    st.header("Employee Predictor")
    st.write("Select an employee profile from our database to predict their risk of leaving and their future performance rating.")
    
    employee_index = st.selectbox("Select Employee by Row Number:", df.index)
    
    if st.button("Run AI Prediction"):
        employee_data = df.iloc[[employee_index]]
        
        # FIX 2: Translate the text back into numbers for the AI Model!
        ml_data = employee_data.copy()
        for col, le in label_encoders.items():
            if col in ml_data.columns:
                ml_data[col] = le.transform(ml_data[col])
        
        # Now feed the numbers to the AI
        X_attrition = ml_data.drop('Attrition', axis=1)
        X_perf = ml_data.drop('PerformanceRating', axis=1)
        
        pred_attrition = attrition_model.predict(X_attrition)
        pred_perf = performance_model.predict(X_perf)
        
        attrition_result = "⚠️ High Risk of Leaving (Yes)" if pred_attrition == 1 else "✅ Likely to Stay (No)"
        
        st.subheader("🔮 AI Prediction Results:")
        col_a, col_p = st.columns(2)
        col_a.success(f"**Attrition Prediction:** {attrition_result}")
        col_p.info(f"**Predicted Performance Rating:** Level {pred_perf}")
        
        st.divider()
        st.write("Here is the employee's background data used for this prediction:")
        st.dataframe(employee_data)