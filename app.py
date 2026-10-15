import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

st.markdown("""
<style>
.main-title {text-align:center; font-size:42px; font-weight:800;}
.sub-title {text-align:center; color:#777; margin-bottom:25px;}
.result-box {padding:20px; border-radius:12px; text-align:center; font-size:25px; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚢 Titanic Survival Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning Mini Project using Logistic Regression</div>', unsafe_allow_html=True)

if not os.path.exists("titanic_model.pkl"):
    st.error("Model not found. First run: python model.py")
    st.stop()

with open("titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Analysis", "ℹ️ About Project"])

with tab1:
    st.subheader("Enter Passenger Details")
    c1, c2 = st.columns(2)

    with c1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3],
                              format_func=lambda x: {1:"1st Class",2:"2nd Class",3:"3rd Class"}[x])
        sex = st.selectbox("Gender", ["male", "female"])
        age = st.slider("Age", 1, 80, 25)
        fare = st.number_input("Ticket Fare", min_value=0.0, value=30.0, step=1.0)

    with c2:
        sibsp = st.number_input("Siblings / Spouses Aboard", 0, 10, 0)
        parch = st.number_input("Parents / Children Aboard", 0, 10, 0)
        embarked = st.selectbox("Embarkation Port", ["S", "C", "Q"],
                                format_func=lambda x: {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[x])

    if st.button("Predict Survival", type="primary", use_container_width=True):
        passenger = pd.DataFrame([{
            "Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp,
            "Parch": parch, "Fare": fare, "Embarked": embarked
        }])

        prediction = model.predict(passenger)[0]
        probability = model.predict_proba(passenger)[0][1]

        if prediction == 1:
            st.success(f"✅ Passenger is predicted to SURVIVE — survival probability: {probability:.1%}")
        else:
            st.error(f"❌ Passenger is predicted NOT TO SURVIVE — survival probability: {probability:.1%}")

with tab2:
    df = pd.read_csv("train.csv")
    st.subheader("Titanic Dataset Overview")
    a, b, c = st.columns(3)
    a.metric("Passengers", len(df))
    b.metric("Survived", int(df["Survived"].sum()))
    c.metric("Did Not Survive", int((df["Survived"] == 0).sum()))

    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Survival Distribution")
    counts = df["Survived"].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(["Did Not Survive", "Survived"], counts.values)
    ax.set_ylabel("Number of Passengers")
    st.pyplot(fig)

    st.subheader("Survival by Gender")
    gender = pd.crosstab(df["Sex"], df["Survived"])
    gender.columns = ["Did Not Survive", "Survived"]
    st.bar_chart(gender)

with tab3:
    st.markdown("""
### Project Objective
Predict whether a Titanic passenger survived using passenger information.

### Machine Learning Algorithm
**Logistic Regression** is used because the output has two classes:
- `0` = Did Not Survive
- `1` = Survived

### Features Used
Passenger Class, Gender, Age, Siblings/Spouses, Parents/Children, Fare and Embarkation Port.

### Project Workflow
Dataset → Preprocessing → Train/Test Split → Logistic Regression → Evaluation → Prediction.
""")
