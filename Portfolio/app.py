import streamlit as st
import boto3
import json

# ----------------------------
# App Title
# ----------------------------
st.title("AAPL 5-Day Return Predictor (SageMaker Deployment)")

# ----------------------------
# SageMaker Endpoint Settings
# ----------------------------
ENDPOINT_NAME = "HW2-pipeline-endpoint-auto"   # MUST match deployed endpoint
AWS_REGION = "us-east-1"

runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

# ----------------------------
# Feature List (ORDER MATTERS)
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

st.header("Enter Feature Values")

inputs = {}

for f in FEATURES:
    if f == "is_quarter_end":
        inputs[f] = float(st.selectbox(f, [0, 1], index=0))
    else:
        inputs[f] = float(st.number_input(f, value=0.0))

row = [inputs[f] for f in FEATURES]

# ----------------------------
# Call SageMaker Endpoint
# ----------------------------
def predict_endpoint(row):
    body = json.dumps([row])

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=body
    )

    result = response["Body"].read().decode("utf-8")

    try:
        parsed = json.loads(result)
        if isinstance(parsed, list):
            return float(parsed[0])
        return float(parsed)
    except:
        return float(result.strip().replace("[", "").replace("]", ""))

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Run Prediction"):
    prediction = predict_endpoint(row)

    st.success(f"Predicted 5-Day Forward Log Return: {prediction:.6f}")

    if prediction > 0:
        st.info("Model expects the stock to go UP 📈")
    else:
        st.info("Model expects the stock to go DOWN 📉")