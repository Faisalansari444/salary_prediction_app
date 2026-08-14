import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- PAGE CONFIGURATION & CSS STYLING ---
st.set_page_config(
    page_title="ZIAF Technologies | Salary Predictor",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
    <style>
    /* Main Background & Font */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    
    /* Metric / Success / Info boxes */
    .stSuccess, .stInfo {
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    
    /* Cards for Layout */
    .css-1r6slb0, .element-container {
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.title("💼 ZIAF Technologies")
st.subheader("Automated Salary Prediction & HR Analytics System")
st.markdown("---")

# Load Dataset
@st.cache_data
def load_data():
    return pd.read_csv("SalaryData.csv")

data = load_data()

# --- SIDEBAR FOR NAVIGATION / INFO ---
with st.sidebar:
    st.image("faizal1.jpg", width=120)
    st.header("About System")
    st.write(
        "This enterprise application utilizes a **Linear Regression** machine learning "
        "model to optimize and automate HR salary benchmarking based on historical employee metrics."
    )
    st.markdown("---")
    st.subheader("🛠️ Tech Stack")
    st.markdown("- **Python & Scikit-Learn**\n- **Pandas & NumPy**\n- **Streamlit UI**")

# --- DATA UNDERSTANDING & METRICS ---
st.header("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", data.shape[0])
col2.metric("Features Tracked", data.shape[1] - 1)
col3.metric("Target Variable", "Salary")

with st.expander("🔍 View Raw Dataset Preview & Statistics"):
    st.write("### First 5 Records")
    st.dataframe(data.head(), use_container_width=True)
    st.write("### Descriptive Statistics")
    st.dataframe(data.describe(), use_container_width=True)

st.markdown("---")

# --- EXPLORATORY DATA ANALYSIS (EDA) ---
st.header("📈 Exploratory Data Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["Salary Distribution", "Experience vs Salary", "Education Levels", "Salary by Job Title"])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data["Salary"], bins=10, color="#3b82f6", edgecolor="black", alpha=0.8)
    ax.set_title("Salary Distribution Frequency")
    ax.set_xlabel("Salary")
    ax.set_ylabel("Employee Count")
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(data["Years of Experience"], data["Salary"], color="#10b981", alpha=0.7)
    ax.set_title("Years of Experience vs Salary")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    st.pyplot(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(8, 4))
    data["Education Level"].value_counts().plot(kind="bar", color="#6366f1", ax=ax, alpha=0.8)
    ax.set_title("Employee Count by Education Level")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    st.pyplot(fig)

with tab4:
    fig, ax = plt.subplots(figsize=(10, 4))
    data.groupby("Job Title")["Salary"].mean().plot(kind="bar", color="#f59e0b", ax=ax, alpha=0.8)
    ax.set_title("Average Salary by Job Title")
    plt.xticks(rotation=30)
    st.pyplot(fig)

st.markdown("---")

# --- DATA PREPROCESSING & MODEL TRAINING ---
data.fillna(data.mean(numeric_only=True), inplace=True)

le_gender = LabelEncoder()
le_education = LabelEncoder()
le_job = LabelEncoder()

data["Gender"] = le_gender.fit_transform(data["Gender"])
data["Education Level"] = le_education.fit_transform(data["Education Level"])
data["Job Title"] = le_job.fit_transform(data["Job Title"])

X = data.drop("Salary", axis=1)
Y = data["Salary"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)
prediction = model.predict(X_test)

# --- MODEL PERFORMANCE ---
st.header("⚙️ Model Performance & Evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("R² Score", f"{r2_score(Y_test, prediction):.4f}")
m2.metric("Mean Absolute Error (MAE)", f"₹ {mean_absolute_error(Y_test, prediction):,.2f}")
m3.metric("Mean Squared Error (MSE)", f"{mean_squared_error(Y_test, prediction):,.2f}")

st.markdown("---")

# --- SALARY PREDICTION INTERFACE ---
st.header("🎯 Live Salary Prediction Tool")
st.write("Provide the candidate profile parameters below to generate an automated salary estimate.")

with st.form("prediction_form"):
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        age = st.number_input("Age", 18, 65, 28)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
        
    with f_col2:
        experience = st.number_input("Years of Experience", 0, 40, 3)
        job = st.selectbox("Job Title", ["Software Engineer", "Data Analyst", "HR", "Manager"])
    
    submitted = st.form_submit_button("Predict Salary Package")

if submitted:
    gender_val = 1 if gender == "Male" else 0
    education_val = {"Bachelor": 0, "Master": 1, "PhD": 2}[education]
    job_val = {"Software Engineer": 0, "Data Analyst": 1, "HR": 2, "Manager": 3}[job]

    input_df = pd.DataFrame([[
        age, gender_val, education_val, job_val, experience
    ]], columns=["Age", "Gender", "Education Level", "Job Title", "Years of Experience"])

    salary = model.predict(input_df)

    if experience < 2:
        category = "Fresher"
    elif experience < 5:
        category = "Junior"
    elif experience < 10:
        category = "Mid-Level"
    else:
        category = "Senior"

    if salary[0] < 400000:
        recommendation = "Suitable for Entry Level Position"
    elif salary[0] < 800000:
        recommendation = "Suitable for Mid-Level Position"
    else:
        recommendation = "Suitable for Senior Position"

    st.markdown("### 📋 Prediction Results")
    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.success(f"**Predicted Salary:**\n### ₹ {salary[0]:,.2f}")
    res_col2.info(f"**Experience Tier:**\n### {category}")
    res_col3.success(f"**HR Recommendation:**\n### {recommendation}")