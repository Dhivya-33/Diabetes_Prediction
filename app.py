# ==============================
# Minimal Diabetes Prediction App
# ==============================

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Load dataset and train model
# ------------------------------
@st.cache_data
def load_model():
    df = pd.read_csv("diabetes.csv")
    cols_with_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    df.fillna(df.mean(), inplace=True)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    svm = SVC(kernel='rbf', C=1, gamma='scale')
    svm.fit(X_train, y_train)
    
    return svm, scaler

svm, scaler = load_model()

# ------------------------------
# 2. Streamlit UI
# ------------------------------
st.title(" Diabetes Prediction")
st.write("Enter patient details to get prediction:")

# Input sliders
pregnancies = st.slider("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose Level", 0, 200, 120)
blood_pressure = st.slider("Blood Pressure", 0, 122, 70)
skin_thickness = st.slider("Skin Thickness", 0, 99, 20)
insulin = st.slider("Insulin Level", 0, 846, 79)
bmi = st.slider("BMI", 0.0, 67.1, 32.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
age = st.slider("Age", 1, 120, 33)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    
    prediction = svm.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error(" The patient **may have diabetes**")
    else:
        st.success("The patient **is unlikely to have diabetes**")
