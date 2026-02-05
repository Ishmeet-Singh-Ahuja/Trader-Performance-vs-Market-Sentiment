import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------- Load Model ----------
model = joblib.load("trader_model.pkl")
features = joblib.load("features.pkl")

st.title("📈 Trader Profitability Predictor")

st.sidebar.header("Input Features")

# ---------- Inputs ----------
Side_Binary = st.sidebar.selectbox("Trade Direction (0=Sell, 1=Buy)", [0, 1])
Size_USD = st.sidebar.number_input("Trade Size USD", value=1000.0)
Trade_Impact = st.sidebar.number_input("Trade Impact", value=0.1)
sentiment_num = st.sidebar.slider("Market Sentiment (0=Fear → 4=Greed)", 0, 4, 2)
pnl_lag1 = st.sidebar.number_input("Previous Trade PnL", value=0.0)
size_lag1 = st.sidebar.number_input("Previous Trade Size", value=1000.0)
sent_lag1 = st.sidebar.slider("Previous Sentiment", 0, 4, 2)

# ---------- Prepare Input ----------
input_data = pd.DataFrame([[
    Side_Binary,
    Size_USD,
    Trade_Impact,
    sentiment_num,
    pnl_lag1,
    size_lag1,
    sent_lag1
]], columns=features)

# ---------- Predict ----------
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    label_map = {0: "🔴 Loss", 1: "⚪ Zero", 2: "🟢 Profit"}

    st.subheader(f"Prediction: {label_map[pred]}")
    st.write("Confidence:", np.round(prob, 3))

    st.bar_chart(pd.DataFrame({
        "Loss": [prob[0]],
        "Zero": [prob[1]],
        "Profit": [prob[2]]
    }))
