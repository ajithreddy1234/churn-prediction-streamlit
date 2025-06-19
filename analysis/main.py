import streamlit as st

import joblib
import numpy as np
scaler=joblib.load(r"C:\Users\venka\PycharmProjects\Churn_analysis\analysis\scaler.pkl")
model=joblib.load(r"C:\Users\venka\PycharmProjects\Churn_analysis\analysis\random_forest_model.pkl")
st.title("Churn Prediction App")
st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction ")
st.divider()
age=st.number_input("Enter age ",min_value=10,max_value=100,value=30)
gender=st.selectbox("Enter the Gender",["Male","Female"])
tenure=st.number_input("Enter Tenure",min_value=0,max_value=130,value=10)
monthlycharge=st.number_input("Enter Monthly Charge ",min_value=30,max_value=150)
st.divider()
predictbutton=st.button("Predict")
if predictbutton:
    gender_selected=1 if gender=="Female" else 0
    x=[age,gender_selected,tenure,monthlycharge]
    x_array=scaler.transform([x])
    prediction=model.predict(x_array)
    predicted="Churn" if prediction==1 else "Not Churn"
    st.balloons()

    st.write(f"Predicted:{predicted}")

else:
    st.write("Please enter the values and use predict button")



