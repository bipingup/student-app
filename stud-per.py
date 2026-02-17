import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------- DARK MODE ----------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        body {background-color: #0E1117; color: white;}
        </style>
    """, unsafe_allow_html=True)

# ---------------- MONGO CONNECTION ----------------
@st.cache_resource
def init_mongo():
    uri = os.getenv("MONGO_URI")

    if not uri:
        return None

    try:
        client = MongoClient(uri, server_api=ServerApi('1'))
        client.admin.command('ping')  # test connection
        db = client["student"]
        return db["student_pred"]
    except Exception as e:
        st.error(f"MongoDB Connection Failed: {e}")
        return None

collection = init_mongo()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("student_lr_final_model.pkl","rb") as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocess(data, scaler, le):
    data['Extracurricular Activities'] = le.transform(
        [data['Extracurricular Activities']]
    )[0]
    df = pd.DataFrame([data])
    return scaler.transform(df)

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center;'>📊 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.write("### Enter student details below")

col1, col2 = st.columns(2)

with col1:
    hours = st.number_input("📘 Hours Studied", 1, 10, 5)
    prev_score = st.number_input("📊 Previous Score", 40, 100, 70)
    extra = st.selectbox("🎯 Extracurricular Activity", ["Yes", "No"])

with col2:
    sleep = st.number_input("😴 Sleep Hours", 4, 10, 7)
    papers = st.number_input("📝 Question Papers Solved", 0, 10, 5)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Performance"):

    try:
        model, scaler, le = load_model()

        user_data = {
            "Hours Studied": hours,
            "Previous Scores": prev_score,
            "Extracurricular Activities": extra,
            "Sleep Hours": sleep,
            "Sample Question Papers Practiced": papers
        }

        processed = preprocess(user_data.copy(), scaler, le)
        prediction = model.predict(processed)
        score = round(float(prediction[0]), 2)

        # -------- RESULT --------
        st.markdown("## 🎯 Prediction Result")
        st.metric("Predicted Score", f"{score}%")

        # -------- GAUGE --------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Performance Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "lightcoral"},
                    {'range': [50, 75], 'color': "khaki"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # -------- FEATURE IMPORTANCE --------
        st.markdown("## 📊 Feature Impact Analysis")

        feature_names = [
            "Hours Studied",
            "Previous Score",
            "Extracurricular",
            "Sleep Hours",
            "Papers Solved"
        ]

        importance = np.array(model.coef_).flatten().tolist()

        fig2 = go.Figure(go.Bar(
            x=importance,
            y=feature_names,
            orientation='h'
        ))

        fig2.update_layout(
            xaxis_title="Impact on Score",
            yaxis_title="Features"
        )

        st.plotly_chart(fig2, use_container_width=True)

        # -------- SAVE TO MONGO --------
        if collection is not None:
            try:
                save_data = user_data.copy()
                save_data["prediction"] = score
                collection.insert_one(save_data)
                st.success("✅ Data stored in MongoDB successfully!")
            except Exception as e:
                st.error(f"Insert failed: {e}")
        else:
            st.warning("⚠ MongoDB not connected")

        # -------- SHOW HISTORY --------
        if collection is not None:
            st.markdown("## 📂 Prediction History")
            history = pd.DataFrame(list(collection.find({}, {"_id": 0})))
            if not history.empty:
                st.dataframe(history, use_container_width=True)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About Project")
st.sidebar.write("""
Machine Learning model built using:
- Linear Regression
- Scikit-learn
- Streamlit
- MongoDB
""")
st.sidebar.info("Developed by Bipin Gupta 🚀")
