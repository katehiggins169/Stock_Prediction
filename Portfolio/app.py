import streamlit as st
import joblib
import numpy as np

# ----------------------------
# App Title
# ----------------------------
st.title("ML Stock Predictor (Local Deployment)")

# ----------------------------
# Load Local Model
# ----------------------------
# This replaces all the AWS/SageMaker code. 
# Make sure 'stock_model.pkl' is uploaded to the 'Portfolio' folder in GitHub.
@st.cache_resource
def load_model():
    return joblib.load('Portfolio/stock_model.pkl')

model = load_model()

# ----------------------------
# Feature List (Updated to match your screenshot)
# ----------------------------
st.header("Inputs")

# Create two columns to match your visual layout
col1, col2 = st.columns(2)

with col1:
    googl = st.number_input("GOOGL", value=0.0)
    dexjpus = st.number_input("DEXJPUS", value=0.0)
    sp500 = st.number_input("SP500", value=0.0)
    vixcls = st.number_input("VIXCLS", value=0.0)

with col2:
    ibm = st.number_input("IBM", value=0.0)
    dexusuk = st.number_input("DEXUSUK", value=0.0)
    djia = st.number_input("DJIA", value=0.0)

# ----------------------------
# Prediction Logic
# ----------------------------
if st.button("Run Prediction"):
    # Organize inputs into a 2D array for the model
    # IMPORTANT: Ensure this order matches how you trained the model in Jupyter!
    row = np.array([[googl, ibm, dexjpus, dexusuk, sp500, djia, vixcls]])
    
    try:
        # Run local prediction
        prediction = model.predict(row)[0]

        st.success(f"Predicted Result: {prediction:.6f}")

        if prediction > 0:
            st.info("Model outlook: POSITIVE 📈")
        else:
            st.info("Model outlook: NEGATIVE 📉")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
