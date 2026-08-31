import streamlit as st
import pandas as pd
import joblib

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Risk Intelligence",
    page_icon="🏢",
    layout="wide"
)


# ============================================================
# 2. LOAD DATA AND MODELS
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_employee_data.csv")


@st.cache_resource
def load_models():
    label_encoders = joblib.load("label_encoders.pkl")
    attrition_model = joblib.load("attrition_model.pkl")
    performance_model = joblib.load("performance_model.pkl")

    return label_encoders, attrition_model, performance_model


df = load_data()

label_encoders, attrition_model, performance_model = load_models()


# ============================================================
# 3. APPLICATION HEADER
# ============================================================

st.title("🏢 Employee Attrition Risk Intelligence")

st.markdown(
    "Analyze workforce trends and identify employee attrition "
    "risk using Machine Learning."
)


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

menu = st.sidebar.radio(
    "Go to:",
    [
        "📊 Overview Dashboard",
        "🤖 Predict Employee Future"
    ]
)


# ============================================================
# SECTION 1 — OVERVIEW DASHBOARD
# ============================================================

if menu == "📊 Overview Dashboard":

    st.header("Workforce Overview")

    # --------------------------------------------------------
    # KEY METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Employees in System",
        len(df)
    )

    attrition_rate = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .mean()
        * 100
    )

    col2.metric(
        "Overall Attrition Rate",
        f"{attrition_rate:.1f}%"
    )

    avg_income = pd.to_numeric(
        df["MonthlyIncome"],
        errors="coerce"
    ).mean()

    col3.metric(
        "Average Monthly Income",
        f"${avg_income:,.0f}"
    )

    st.divider()

    # --------------------------------------------------------
    # ATTRITION BY DEPARTMENT
    # --------------------------------------------------------

    st.subheader("Attrition Counts by Department")

    st.write(
        "This chart shows the number of employees who left "
        "from each department."
    )

    dept_attrition = (
        df[
            df["Attrition"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
        ]
        .groupby("Department")
        .size()
        .sort_values(ascending=True)
    )

    st.bar_chart(dept_attrition)

    st.divider()

    # --------------------------------------------------------
    # ATTRITION BY JOB ROLE
    # --------------------------------------------------------

    st.subheader("Attrition by Job Role")

    st.write(
        "This chart identifies job roles with higher numbers "
        "of employee departures."
    )

    role_attrition = (
        df[
            df["Attrition"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
        ]
        .groupby("JobRole")
        .size()
        .sort_values(ascending=True)
    )

    st.bar_chart(role_attrition)


# ============================================================
# SECTION 2 — MACHINE LEARNING PREDICTOR
# ============================================================

elif menu == "🤖 Predict Employee Future":

    st.header("🤖 Employee Attrition Risk Predictor")

    st.write(
        "Enter employee details to estimate attrition risk "
        "using our Machine Learning model."
    )

    st.subheader("Employee Information")

    # --------------------------------------------------------
    # EMPLOYEE INPUTS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            18,
            60,
            30
        )

        business_travel = st.selectbox(
            "Business Travel",
            [
                "Non-Travel",
                "Travel_Frequently",
                "Travel_Rarely"
            ]
        )

        department = st.selectbox(
            "Department",
            [
                "Human Resources",
                "Research & Development",
                "Sales"
            ]
        )

        distance_home = st.number_input(
            "Distance From Home",
            1,
            30,
            5
        )

        education = st.slider(
            "Education",
            1,
            5,
            3
        )

        education_field = st.selectbox(
            "Education Field",
            [
                "Human Resources",
                "Life Sciences",
                "Marketing",
                "Medical",
                "Other",
                "Technical Degree"
            ]
        )

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )

        job_involvement = st.slider(
            "Job Involvement",
            1,
            4,
            3
        )

        job_level = st.slider(
            "Job Level",
            1,
            5,
            2
        )

        job_role = st.selectbox(
            "Job Role",
            [
                "Healthcare Representative",
                "Human Resources",
                "Laboratory Technician",
                "Manager",
                "Manufacturing Director",
                "Research Director",
                "Research Scientist",
                "Sales Executive",
                "Sales Representative"
            ]
        )

    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        job_satisfaction = st.slider(
            "Job Satisfaction",
            1,
            4,
            3
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "Divorced",
                "Married",
                "Single"
            ]
        )

        monthly_income = st.number_input(
            "Monthly Income",
            1000,
            20000,
            5000
        )

        num_companies = st.number_input(
            "Number of Companies Worked",
            0,
            20,
            2
        )

        overtime = st.selectbox(
            "OverTime",
            [
                "No",
                "Yes"
            ]
        )

        percent_salary_hike = st.slider(
            "Percent Salary Hike",
            10,
            30,
            15
        )

        relationship_satisfaction = st.slider(
            "Relationship Satisfaction",
            1,
            4,
            3
        )

        stock_option = st.slider(
            "Stock Option Level",
            0,
            3,
            1
        )

        total_working_years = st.number_input(
            "Total Working Years",
            0,
            40,
            8
        )

        training_times = st.slider(
            "Training Times Last Year",
            0,
            10,
            3
        )

    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        environment_satisfaction = st.slider(
            "Environment Satisfaction",
            1,
            4,
            3
        )

        daily_rate = st.number_input(
            "Daily Rate",
            100,
            1500,
            800
        )

        hourly_rate = st.number_input(
            "Hourly Rate",
            30,
            100,
            60
        )

        monthly_rate = st.number_input(
            "Monthly Rate",
            2000,
            30000,
            14000
        )

        performance_rating = st.slider(
            "Performance Rating",
            1,
            4,
            3
        )

        work_life_balance = st.slider(
            "Work Life Balance",
            1,
            4,
            3
        )

        years_at_company = st.number_input(
            "Years At Company",
            0,
            40,
            5
        )

        years_current_role = st.number_input(
            "Years In Current Role",
            0,
            20,
            3
        )

        years_since_promotion = st.number_input(
            "Years Since Last Promotion",
            0,
            15,
            1
        )

        years_manager = st.number_input(
            "Years With Current Manager",
            0,
            20,
            3
        )

    st.divider()

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Employee Risk",
        type="primary"
    ):

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "Age": age,

            "BusinessTravel": business_travel,

            "DailyRate": daily_rate,

            "Department": department,

            "DistanceFromHome": distance_home,

            "Education": education,

            "EducationField": education_field,

            "EnvironmentSatisfaction": environment_satisfaction,

            "Gender": gender,

            "HourlyRate": hourly_rate,

            "JobInvolvement": job_involvement,

            "JobLevel": job_level,

            "JobRole": job_role,

            "JobSatisfaction": job_satisfaction,

            "MaritalStatus": marital_status,

            "MonthlyIncome": monthly_income,

            "MonthlyRate": monthly_rate,

            "NumCompaniesWorked": num_companies,

            "OverTime": overtime,

            "PercentSalaryHike": percent_salary_hike,

            "PerformanceRating": performance_rating,

            "RelationshipSatisfaction": relationship_satisfaction,

            "StockOptionLevel": stock_option,

            "TotalWorkingYears": total_working_years,

            "TrainingTimesLastYear": training_times,

            "WorkLifeBalance": work_life_balance,

            "YearsAtCompany": years_at_company,

            "YearsInCurrentRole": years_current_role,

            "YearsSinceLastPromotion": years_since_promotion,

            "YearsWithCurrManager": years_manager

        }])

        # ----------------------------------------------------
        # ENCODE CATEGORICAL DATA
        # ----------------------------------------------------

        ml_data = input_data.copy()

        for col, le in label_encoders.items():

            if col in ml_data.columns:

                ml_data[col] = le.transform(
                    ml_data[col]
                )

        # ----------------------------------------------------
        # ATTRITION MODEL
        # ----------------------------------------------------

        ml_data = ml_data[
            attrition_model.feature_names_in_
        ]

        pred_attrition = attrition_model.predict(
            ml_data
        )

        # ----------------------------------------------------
        # ATTRITION PROBABILITY
        # ----------------------------------------------------

        if hasattr(
            attrition_model,
            "predict_proba"
        ):

            risk_probability = (
                attrition_model
                .predict_proba(ml_data)[0][1]
                * 100
            )

        else:

            risk_probability = None

        # ----------------------------------------------------
        # PERFORMANCE MODEL
        # ----------------------------------------------------

        X_perf = ml_data.copy()

        X_perf["Attrition"] = pred_attrition[0]

        X_perf = X_perf[
            performance_model.feature_names_in_
        ]

        pred_perf = performance_model.predict(
            X_perf
        )

        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.subheader(
            "🔮 AI Prediction Results"
        )

        col_a, col_p = st.columns(2)

        # ----------------------------------------------------
        # ATTRITION RESULT
        # ----------------------------------------------------

        if pred_attrition[0] == 1:

            col_a.error(
                "⚠️ HIGH ATTRITION RISK"
            )

        else:

            col_a.success(
                "✅ LOW ATTRITION RISK"
            )

        # ----------------------------------------------------
        # PERFORMANCE RESULT
        # ----------------------------------------------------

        col_p.info(
            f"⭐ Predicted Performance Rating: "
            f"{pred_perf[0]}"
        )

        # ====================================================
        # RISK ANALYSIS
        # ====================================================

        if risk_probability is not None:

            st.metric(
                "Estimated Probability of Leaving",
                f"{risk_probability:.1f}%"
            )

            st.progress(
                min(
                    risk_probability / 100,
                    1.0
                )
            )

            # ------------------------------------------------
            # LOW RISK
            # ------------------------------------------------

            if risk_probability < 30:

                st.success(
                    "🟢 LOW RISK — Employee is less likely to leave."
                )

                st.info(
                    "💡 HR Recommendation: Continue regular "
                    "employee engagement and monitor performance "
                    "periodically."
                )

            # ------------------------------------------------
            # MEDIUM RISK
            # ------------------------------------------------

            elif risk_probability < 60:

                st.warning(
                    "🟡 MEDIUM RISK — Employee shows moderate "
                    "attrition risk."
                )

                st.info(
                    "💡 HR Recommendation: Consider an employee "
                    "engagement discussion and review workload, "
                    "satisfaction and career growth opportunities."
                )

            # ------------------------------------------------
            # HIGH RISK
            # ------------------------------------------------

            else:

                st.error(
                    "🔴 HIGH RISK — Employee has a high "
                    "predicted probability of leaving."
                )

                st.info(
                    "💡 HR Recommendation: Prioritize this employee "
                    "for HR intervention. Review overtime, job "
                    "satisfaction, work-life balance, compensation "
                    "and career growth."
                )

        # ====================================================
        # EMPLOYEE PROFILE
        # ====================================================

        st.subheader(
            "👤 Employee Profile"
        )

        st.dataframe(
            input_data,
            use_container_width=True
        )