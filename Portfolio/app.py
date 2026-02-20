import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ----------------------------
# 1. App Configuration
# ----------------------------
st.set_page_config(page_title="Stock Prediction", layout="wide")
st.title("📈 Stock Return Predictor")
st.markdown("Enter the feature values from your Jupyter analysis below to predict returns.")

# ----------------------------
# 2. Load the Local Model
# ----------------------------
@st.cache_resource
def load_my_model():
    # Make sure 'stock_prediction_model.pkl' is in your GitHub Portfolio folder
    try:
        return joblib.load('Portfolio/stock_prediction_model.pkl')
    except:
        # Fallback if the file is in the root directory instead
        return joblib.load('stock_prediction_model.pkl')

model = load_my_model()

# ----------------------------
# 3. User Inputs (Matching your Jupyter Features)
# ----------------------------
st.header("Model Features")

col1, col2 = st.columns(2)

with col1:
    googl = st.number_input("GOOGL Price/Return", value=0.0, format="%.6f")
    ibm = st.number_input("IBM Price/Return", value=0.0, format="%.6f")
    dexjpus = st.number_input("DEXJPUS (Exchange Rate)", value=0.0, format="%.6f")
    dexusuk = st.number_input("DEXUSUK (Exchange Rate)", value=0.0, format="%.6f")

with col2:
    sp500 = st.number_input("SP500 Index Value", value=0.0, format="%.6f")
    djia = st.number_input("DJIA Index Value", value=0.0, format="%.6f")
    vixcls = st.number_input("VIXCLS (Volatility)", value=0.0, format="%.6f")

# ----------------------------
# 4. Prediction Execution
# ----------------------------
if st.button("Run Prediction", type="primary"):
    # ORDER MATTERS: Must match the order used during model.fit() in Jupyter
    feature_values = np.array([[googl, ibm, dexjpus, dexusuk, sp500, djia, vixcls]])
    
    try:
        prediction = model.predict(feature_values)[0]
        
        # Display Result
        st.divider()
        st.subheader("Results")
        
        if prediction > 0:
            st.success(f"**Predicted Return: {prediction:.6f}**")
            st.info("Market Sentiment: **BULLISH** 📈")
        else:
            st.error(f"**Predicted Return: {prediction:.6f}**")
            st.info("Market Sentiment: **BEARISH** 📉")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.warning("Check if the number of features matches what the model expects.")
