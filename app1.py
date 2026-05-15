import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Title
# -----------------------------
st.title("Logistic Classification App")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("college_student_placement_dataset.csv")

st.subheader("Dataset")
st.write(df.head())

# -----------------------------
# Preprocessing
# -----------------------------

# Drop unnecessary column
df = df.drop("College_ID", axis=1)

# Label Encoding
le1 = LabelEncoder()
df["Internship_Experience"] = le1.fit_transform(df["Internship_Experience"])

le2 = LabelEncoder()
df["Placement"] = le2.fit_transform(df["Placement"])

# Features and Target
X = df.drop("Placement", axis=1)
y = df["Placement"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Logistic Regression Model
# -----------------------------
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy Score: {accuracy:.2f}")

# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter Student Details")

IQ = st.number_input("IQ Score", min_value=0)

CGPA = st.number_input("CGPA", min_value=0.0)

Academic_Performance = st.number_input(
    "Academic Performance",
    min_value=0.0
)

Communication_Skills = st.number_input(
    "Communication Skills",
    min_value=0
)

Internship_Experience = st.selectbox(
    "Internship Experience",
    ["No", "Yes"]
)

Projects_Completed = st.number_input(
    "Projects Completed",
    min_value=0
)

Extra_Curricular_Score = st.number_input(
    "Extra Curricular Score",
    min_value=0
)

Placement_Test_Score = st.number_input(
    "Placement Test Score",
    min_value=0
)

# Encode Internship Experience
internship_encoded = le1.transform([Internship_Experience])[0]

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict"):

    input_data = [[
        IQ,
        CGPA,
        Academic_Performance,
        Communication_Skills,
        internship_encoded,
        Projects_Completed,
        Extra_Curricular_Score,
        Placement_Test_Score
    ]]

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Output
    if prediction[0] == 1:
        st.success("Student is Likely to be Placed")
    else:
        st.error("Student is Not Likely to be Placed")

# -----------------------------
# Classification Report
# -----------------------------
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.write(report_df)