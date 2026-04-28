import os
import warnings
import json
import tempfile
import posixpath

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

import joblib
import tarfile
import shap

warnings.simplefilter("ignore")

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("Fraud Detection Prediction App")

# -----------------------------
# Load template row
# -----------------------------
file_path = "X_train_template.csv"

dataset = pd.read_csv(file_path)
dataset = dataset.loc[:, ~dataset.columns.str.contains("^Unnamed")]

# -----------------------------
# AWS secrets
# -----------------------------
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# -----------------------------
# AWS session
# -----------------------------
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# -----------------------------
# Model info
# -----------------------------
MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "fraud_shap_explainer.joblib",
    "pipeline": "model.tar.gz",
    "keys": ["TransactionAmt", "addr1", "addr2"],
    "inputs": [
        {"name": "TransactionAmt", "type": "number", "min": 0.0, "max": 10000.0, "default": 100.0, "step": 1.0},
        {"name": "addr1", "type": "number", "min": 0.0, "max": 1000.0, "default": 300.0, "step": 1.0},
        {"name": "addr2", "type": "number", "min": 0.0, "max": 1000.0, "default": 87.0, "step": 1.0}
    ]
}

# -----------------------------
# Load pipeline from S3 for SHAP
# -----------------------------
@st.cache_resource
def load_pipeline(_session, bucket, s3_folder):
    s3_client = _session.client("s3")
    filename = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Bucket=bucket,
        Key=f"{s3_folder}/{filename}",
        Filename=filename
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_files = [f for f in tar.getnames() if f.endswith(".joblib")]

    if len(joblib_files) == 0:
        raise FileNotFoundError("No .joblib model file found inside model.tar.gz")

    return joblib.load(joblib_files[0])

# -----------------------------
# Load SHAP explainer from S3
# -----------------------------
@st.cache_resource
def load_shap_explainer(_session, bucket, s3_key, local_path):
    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(
            Bucket=bucket,
            Key=s3_key,
            Filename=local_path
        )

    return joblib.load(local_path)

# -----------------------------
# Call SageMaker endpoint
# -----------------------------
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    try:
        payload = input_df.to_dict(orient="records")
        raw_pred = predictor.predict(payload)

        if isinstance(raw_pred, list):
            pred_val = raw_pred[0]
        elif isinstance(raw_pred, dict) and "predictions" in raw_pred:
            pred_val = raw_pred["predictions"][0]
        else:
            pred_val = raw_pred

        pred_val = int(pred_val)

        mapping = {
            0: "Legitimate",
            1: "Fraud"
        }

        return mapping.get(pred_val, str(pred_val)), 200

    except Exception as e:
        return f"Error: {str(e)}", 500

# -----------------------------
# SHAP explanation
# -----------------------------
def display_explanation(input_df):
    try:
        explainer_name = MODEL_INFO["explainer"]

        explainer = load_shap_explainer(
            session,
            aws_bucket,
            posixpath.join("explainer", explainer_name),
            os.path.join(tempfile.gettempdir(), explainer_name)
        )

        best_pipeline = load_pipeline(session, aws_bucket, "fraud-model")

        # Use all steps before sampler/selector/to_dense/model if possible
        preprocessing_steps = best_pipeline.steps[:-4]
        preprocessing_pipeline = type(best_pipeline)(steps=preprocessing_steps)

        transformed = preprocessing_pipeline.transform(input_df)

        if hasattr(best_pipeline.named_steps.get("preprocess"), "get_feature_names_out"):
            try:
                feature_names = best_pipeline.named_steps["preprocess"].get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
        else:
            feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

        selected = best_pipeline.named_steps["selector"].transform(
            best_pipeline.named_steps["preprocess"].transform(transformed)
        )

        if "to_dense" in best_pipeline.named_steps:
            selected = best_pipeline.named_steps["to_dense"].transform(selected)

        shap_values = explainer.shap_values(selected)

        st.subheader("Decision Transparency: SHAP")

        fig = plt.figure(figsize=(10, 4))

        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], selected, show=False)
        else:
            shap.summary_plot(shap_values, selected, show=False)

        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP explanation could not be displayed: {e}")

# -----------------------------
# Streamlit UI
# -----------------------------
st.subheader("Enter Transaction Information")

with st.form("pred_form"):
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"],
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"]
            )

    submitted = st.form_submit_button("Run Prediction")

# Build full model input row from template
input_row = dataset.iloc[0:1].copy()

for key, value in user_inputs.items():
    if key in input_row.columns:
        input_row[key] = value

if submitted:
    st.subheader("Prediction Result")

    res, status = call_model_api(input_row)

    if status == 200:
        if res == "Fraud":
            st.error(f"Prediction: {res}")
        else:
            st.success(f"Prediction: {res}")

        st.write("Inputs sent to model:")
        st.dataframe(input_row[MODEL_INFO["keys"]])

        display_explanation(input_row)

    else:
        st.error(res)
