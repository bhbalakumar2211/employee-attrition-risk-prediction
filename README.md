# Employee Attrition Analysis & Prediction

### HR Analytics | Machine Learning | Predictive Analytics | Streamlit

> An end-to-end HR analytics and machine learning project that analyzes employee attrition patterns, identifies workforce trends, predicts attrition risk and performance rating, and delivers interactive insights through a Streamlit application.

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-2F7D32)](https://joblib.readthedocs.io/)

---

## Project Overview

Employee attrition can create significant business costs through recruitment, onboarding, training, productivity loss, and workforce disruption.

This project treats employee attrition as a **data analytics and machine learning problem**, combining:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Workforce KPI analysis
- Categorical feature encoding
- Employee attrition classification
- Performance rating prediction
- Model evaluation
- Model persistence using Joblib
- Interactive Streamlit visualization
- Employee-level prediction

The goal is to demonstrate how workforce data can be transformed into **actionable HR insights and predictive decision support**.

---

## Business Problem

HR teams need to understand not only **what happened**, but also **which workforce patterns may be associated with employee turnover**.

This project addresses questions such as:

- What is the overall employee attrition rate?
- Which departments experience higher attrition?
- How does overtime relate to employee turnover?
- How does job satisfaction vary across the workforce?
- Which employee profiles may have higher predicted attrition risk?
- Can machine learning support proactive employee-retention analysis?

---

## Project Objectives

### Primary Objective

Build an end-to-end HR analytics solution that combines descriptive analytics with machine learning-based prediction.

### Key Objectives

- Clean and prepare employee-level data.
- Explore workforce and attrition patterns.
- Identify relevant HR analytics dimensions.
- Encode categorical variables for machine learning.
- Train an employee attrition classification model.
- Train a performance-rating classification model.
- Evaluate model performance using classification metrics.
- Persist trained models and encoders.
- Integrate the models into an interactive Streamlit application.

---

## Dataset

The dataset contains:

- **1,470 employee records**
- **35 original features**
- Demographic attributes
- Department and job information
- Compensation-related attributes
- Job satisfaction measures
- Work-life balance indicators
- Tenure and experience variables
- Performance-related variables
- Employee attrition information

### Data Preprocessing

The following constant / non-analytical columns were removed:

```text
EmployeeCount
EmployeeNumber
Over18
StandardHours

---

## Project Objectives

### Primary Objective

Build an end-to-end HR analytics solution capable of analyzing employee attrition and generating predictive insights.

### Secondary Objectives

* Clean and prepare employee-level data.
* Identify workforce patterns through EDA.
* Encode categorical variables for machine learning.
* Train an attrition classification model.
* Train a performance-rating classification model.
* Evaluate model performance.
* Persist trained models using Joblib.
* Integrate the models into a Streamlit application.

---

## Dataset

The dataset contains:

* **1,470 employee records**
* **35 original features**
* Demographic information
* Department and job information
* Compensation attributes
* Job satisfaction measures
* Work-life balance indicators
* Tenure and experience variables
* Performance-related variables
* Attrition target

### Data preprocessing

The following constant / non-analytical columns were removed:

```text
EmployeeCount
EmployeeNumber
Over18
StandardHours
```

Dashboard Preview
Workforce Analytics

Attrition Analysis

Employee Prediction

---

## End-to-End Workflow

```text
                             RAW EMPLOYEE DATA
                         │
                         ▼
                 DATA CLEANING
                         │
                         ▼
                DATA PREPARATION
                         │
                         ▼
                  EDA / ANALYSIS
                         │
                         ▼
             CATEGORICAL ENCODING
                         │
                         ▼
                 TRAIN / TEST SPLIT
                         │
                         ▼
              RANDOM FOREST MODELS
                    ┌────┴────┐
                    ▼         ▼
              ATTRITION   PERFORMANCE
              PREDICTION   PREDICTION
                    │         │
                    └────┬────┘
                         ▼
                MODEL PERSISTENCE
                     Joblib
                         │
                         ▼
                STREAMLIT APPLICATION
                         │
                         ▼
             HR ANALYTICS + PREDICTION
```

---

Machine Learning
1. Employee Attrition Prediction
Target Variable
Attrition

The model predicts:

0 → Employee stays
1 → Employee leaves
Algorithm
Random Forest Classifier
Training Configuration
n_estimators = 100
random_state = 42
test_size = 20%
Current Model Results
Metric	Result
Accuracy	88.10%
Precision — Attrition Class	0.83
Recall — Attrition Class	0.13
F1-score — Attrition Class	0.22
Model Interpretation

The 88.10% accuracy should not be interpreted as production-level predictive performance.

The dataset contains an imbalance between employees who stayed and employees who left. As a result, the model achieves relatively high overall accuracy while detecting only a small proportion of actual attrition cases.

For an HR retention use case, minority-class recall is particularly important, because failing to identify an employee who is genuinely at risk may be more costly than a small reduction in overall accuracy.

Therefore, the current model is presented as a portfolio-level machine learning implementation, not as a production-ready HR decision engine.

2. Performance Rating Prediction
Target Variable
PerformanceRating
Algorithm
Random Forest Classifier
Reported Accuracy
100%
Important Validation Note

The current training run reports 100% accuracy for performance-rating prediction.

This result requires further investigation before being interpreted as genuine predictive capability.

Areas requiring additional validation include:

Potential feature leakage
Target separability
Dataset construction
Cross-validation performance
Generalization to unseen data

The project therefore does not present the 100% score as evidence of production readiness.

Exploratory Data Analysis

The project analyzes employee attrition and workforce patterns across multiple dimensions, including:

Overall attrition
Department
Age
Gender
Overtime
Job role
Job satisfaction
Workforce distribution
Compensation-related attributes
Tenure and experience

The Streamlit application converts these analytical findings into interactive workforce visualizations.

Streamlit Application

The application provides two major analytical workflows.

Workforce Overview

The dashboard provides:

Total employees
Overall attrition rate
Average monthly income
Attrition by department
Attrition by gender
Attrition by overtime
Attrition by job role
Employee distribution by department
Job satisfaction distribution
Employee Prediction

The prediction workflow uses the persisted model and encoders to generate:

Predicted attrition status
Attrition probability
Predicted performance rating
Employee-level prediction output
Technology Stack
Programming
Python
Data Analysis
Pandas
NumPy
Matplotlib
Seaborn
Machine Learning
Scikit-learn
Random Forest Classifier
Label Encoding
Model Persistence
Joblib
Application
Streamlit
Repository Structure
Employee-Attrition-Analysis-and-Prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
│
├── 1_data_analysis.ipynb
│       └── Data inspection, cleaning and EDA
│
├── 2_model_training.ipynb
│       └── Feature encoding, model training and evaluation
│
├── employee_data.csv
│       └── Original employee dataset
│
├── cleaned_employee_data.csv
│       └── Cleaned analytical dataset
│
├── attrition_model.pkl
│       └── Trained attrition model
│
├── performance_model.pkl
│       └── Trained performance model
│
├── label_encoders.pkl
│       └── Persisted categorical encoders
│
└── screenshots/
    ├── dashboard.png
    ├── attrition-analysis.png
    └── prediction.png
How to Run Locally
1. Clone the Repository
git clone https://github.com/Giftson-22/Employee-Attrition-Analysis-and-Prediction.git
cd Employee-Attrition-Analysis-and-Prediction
2. Create a Virtual Environment
python -m venv .venv
macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate
3. Install Dependencies

Install the required Python packages using:

pip install -r requirements.txt
4. Run the Streamlit Application
streamlit run app.py

The application will open in your browser.

Business Value

The project demonstrates how HR analytics can support:

Retention Risk Analysis

Identify employee profiles associated with higher predicted attrition risk.

Workforce Analysis

Understand attrition patterns across departments, job roles, overtime, gender and satisfaction.

Retention Planning

Use workforce insights to identify areas that may require further HR investigation.

Data-Driven Decision Support

Move beyond historical reporting toward predictive workforce analytics.

Important: Model predictions should be treated as decision-support signals rather than automated HR decisions.

Model Limitations

This project is a portfolio-level analytical and machine learning solution, not a production HR decision system.

Current limitations include:

Attrition class imbalance.
Low recall for the attrition class.
No ROC-AUC currently reported in the training notebook.
No confusion matrix currently implemented.
No stratified cross-validation.
Direct LabelEncoder usage instead of a reusable preprocessing pipeline.
Performance model requires further validation because of its reported 100% accuracy.
Prediction currently operates on an existing employee record rather than a fully independent employee-profile form.
Some application findings are currently hard-coded rather than dynamically calculated.
The current model evaluation is based on a single train/test split.

These limitations are documented intentionally because high accuracy alone does not prove that a model generalizes well or provides useful business predictions.

Planned Improvements
Machine Learning
 Add stratified cross-validation.
 Address class imbalance using class weights and/or resampling.
 Compare Logistic Regression, Random Forest and Gradient Boosting.
 Perform hyperparameter tuning.
 Add ROC-AUC.
 Add PR-AUC.
 Add confusion matrix.
 Optimize attrition-class recall and F1-score.
Explainability
 Add feature importance analysis.
 Add SHAP-based explanations.
 Explain individual employee risk predictions.
Application
 Replace existing-row selection with an independent employee-profile input form.
 Add dynamic filtering.
 Add model performance visualizations.
 Add richer workforce analytics.
 Add downloadable analytical reports.
Engineering
 Add automated tests.
 Introduce Pipeline and ColumnTransformer.
 Separate preprocessing, model training and application logic.
 Add Docker support.
 Deploy the application to the cloud.
 Improve reproducibility and model versioning.
What This Project Demonstrates
Data Analytics
Data cleaning
Exploratory Data Analysis
Workforce KPI analysis
Business insight generation
HR analytics
Data Science
Classification
Feature preprocessing
Train/test validation
Random Forest
Model persistence
Prediction serving
Application Development
Streamlit
Model integration
Interactive analytics
End-to-end application workflow
Business Thinking

The project connects technical analysis to a practical business problem:

Employee Data
      ↓
Workforce Analysis
      ↓
Attrition Patterns
      ↓
Risk Prediction
      ↓
Retention Decision Support
Key Takeaway

This project demonstrates the ability to move from:

Business Problem → Data → Analysis → Machine Learning → Application

The project also highlights an important principle of applied machine learning:

A high accuracy score is not enough. Model performance must be evaluated against the actual business objective.

In this case, the current attrition model's low recall demonstrates why class imbalance, appropriate evaluation metrics, validation strategy and business context matter when developing predictive HR analytics solutions.

Future Direction

The next iteration of the project will focus on:

Improving minority-class recall
Robust cross-validation
Better model comparison
Explainable AI
Independent employee-profile prediction
Reproducible preprocessing pipelines
Automated testing
Production-oriented deployment


Author
Giftson Pratap Singh
Data Analytics | Data Science | HR Analytics
📍 Chennai, India.
LinkedIn: https://www.linkedin.com/in/giftson-pratap-singh-3b580061
GitHub: https://github.com/Giftson-22

⭐ If you find this project useful, consider starring the repository.
