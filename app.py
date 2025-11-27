import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

model = pickle.load(open("regmodel.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    h1 {
        color: #2C3E50;
        text-align: center;
    }
    .block-container {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        font-size: 18px;
        height: 50px;
        width: 100%;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #1A5276;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🏠 Boston House Price Prediction</h1>", unsafe_allow_html=True)
st.write("### Enter the house details below using sliders:")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    CRIM = st.slider("Crime Rate (CRIM)", 0.0, 100.0, 0.5)
    ZN = st.slider("Residential Land (ZN)", 0.0, 100.0, 10.0)
    INDUS = st.slider("Industrial Area (INDUS)", 0.0, 30.0, 5.0)
    CHAS = st.selectbox("Near Charles River (CHAS)", [0, 1])
    NOX = st.slider("Nitric Oxide Level (NOX)", 0.0, 1.0, 0.5)
    RM = st.slider("Number of Rooms (RM)", 1.0, 10.0, 6.0)
    AGE = st.slider("Age of Property (AGE)", 0.0, 100.0, 50.0)

with col2:
    DIS = st.slider("Distance to Work Centres (DIS)", 0.0, 15.0, 5.0)
    RAD = st.slider("Accessibility to Highways (RAD)", 1, 25, 5)
    TAX = st.slider("Property Tax (TAX)", 100, 800, 300)
    PTRATIO = st.slider("Pupil-Teacher Ratio", 10.0, 25.0, 15.0)
    B = st.slider("B Value", 0.0, 400.0, 300.0)
    LSTAT = st.slider("Lower Status (%)", 0.0, 40.0, 10.0)

st.markdown("---")

if st.button("🔍 Predict House Price"):

    input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX,
                            RM, AGE, DIS, RAD, TAX,
                            PTRATIO, B, LSTAT]])

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]

    st.markdown(f"""
        <div style="
            background-color:#D4E6F1;
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:25px;
            color:#154360;
            ">
            💰 <b>Predicted House Price:</b> ${pred * 1000:,.2f}
        </div>
    """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("<center>Made by Soumya using Machine Learning & Streamlit</center>", unsafe_allow_html=True)
