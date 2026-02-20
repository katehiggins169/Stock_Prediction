import os
import json
import boto3
import streamlit as st

st.set_page_config(page_title="HW2 Stock Predictor", layout="centered")
st.title("HW2: AAPL 5-Day Return Predictor")

# ----------------------------
# Config (use Streamlit Secrets / env vars when deployed)
# ----------------------------
ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "HW2-pipeline-endpoint-auto")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

# ----------------------------
# Features (MUST match training order)
# ----------------------------
FEATURES = [
    'MSFT',
    'SPY',
    'DEXUSUK',
    'SP500',
    'VIXCLS',
    'AAPL_mom_5',
    'AAPL_mom_10',
    'AAPL_ma10_gap',
    'AAPL_ma20_gap',
    'AAPL_hl_spread',
    'AAPL_oc_return',
    'is_quarter_end'
]

st.caption(f"Endpoint: {ENDPOINT_NAME} | Region: {AWS_REGION}")
st.subheader("Inputs")

inputs = {}
for f in FEATURES:
    if f == "is_quarter_end":
        inputs[f] = float(st.selectbox(f, [0, 1], index=0))
    else:
        inputs[f] = float(st.number_input(f, value=0.0))

row = [inputs[f] for f in FEATURES]

def predict_endpoint(row):
    body = json.dumps([row])  # one-row batch
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=body
    )
    raw = resp["Body"].read().decode("utf-8")

    # Parse common response formats
    try:
        pred = json.loads(raw)
        if isinstance(pred, list):
            return float(pred[0] if not isinstance(pred[0], list) else pred[0][0])
        return float(pred)
    except Exception:
        return float(raw.strip().replace("[", "").replace("]", ""))

if st.button("Run Prediction"):
    try:
        pred = predict_endpoint(row)
        st.success(f"Predicted 5-day forward log return: {pred:.6f}")
        st.info("Direction: " + ("UP 📈" if pred > 0 else "DOWN 📉"))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.warning("Check AWS credentials / region / endpoint name in Streamlit Secrets.")
