import os
import warnings
import tempfile
import posixpath
import tarfile
import sys

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

import joblib
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
file_path = "Portfolio/X_train.csv"

try:
    dataset = pd.read_csv(file_path)
    dataset = dataset.loc[:, ~dataset.columns.str.contains("^Unnamed")]
except Exception as e:
    st.error(f"Could not load X_train.csv: {e}")
    st.stop()

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

    "keys": ["TransactionAmt", "card1", "D2", "D10", "C1"],

    "inputs": [
        {"name": "TransactionAmt", "label": "Transaction Amount ($)", "min": 0.0, "max": 10000.0, "default": 100.0, "step": 1.0},
        {"name": "card1", "label": "Card ID (Anonymized)", "min": 0.0, "max": 20000.0, "default": 10000.0, "step": 1.0},
        {"name": "D2", "label": "Time Since Last Transaction", "min": 0.0, "max": 10000.0, "default": 100.0, "step": 1.0},
        {"name": "D10", "label": "Account Activity Timing", "min": 0.0, "max": 10000.0, "default": 100.0, "step": 1.0},
        {"name": "C1", "label": "Transaction Frequency Pattern", "min": 0.0, "max": 5000.0, "default": 1.0, "step": 1.0},
    ]
}

# -----------------------------
# Load pipeline from S3 for SHAP
# -----------------------------
@st.cache_resource
def load_pipeline(_session, bucket, s3_folder):
    s3_client = _session.client("s3")
    local_filename = os.path.join(tempfile.gettempdir(), MODEL_INFO["pipeline"])

    s3_client.download_file(
        Bucket=bucket,
        Key=f"{s3_folder}/{MODEL_INFO['pipeline']}",
        Filename=local_filename
    )

    extract_dir = tempfile.mkdtemp()

    with tarfile.open(local_filename, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_files = [
            os.path.join(extract_dir, f)
            for f in tar.getnames()
            if f.endswith(".joblib")
        ]

    # ADD THESE TWO LINES
    if extract_dir not in sys.path:
        sys.path.insert(0, extract_dir)

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
    runtime = session.client("sagemaker-runtime")

    try:
        payload = input_df.to_csv(index=False)

        response = runtime.invoke_endpoint(
            EndpointName=MODEL_INFO["endpoint"],
            ContentType="text/csv",
            Body=payload
        )

        result = response["Body"].read().decode("utf-8").strip()

        # clean result
        result = result.replace("[", "").replace("]", "").replace('"', "").strip()
        pred_val = int(float(result))

        if pred_val == 1:
            return "Fraud", 200
        else:
            return "Legitimate", 200

    except Exception as e:
        return f"Prediction error: {str(e)}", 500

# -----------------------------
# SHAP explanation
# -----------------------------
def display_explanation(input_df):
    try:
        best_pipeline = load_pipeline(session, aws_bucket, "fraud-model")

        X_eng = best_pipeline.named_steps["feature_builder"].transform(input_df.copy())
        X_eng = best_pipeline.named_steps["drop_missing"].transform(X_eng)
        X_eng = best_pipeline.named_steps["drop_constant"].transform(X_eng)
        X_eng = best_pipeline.named_steps["group_rare"].transform(X_eng)
        X_eng = best_pipeline.named_steps["drop_high_card"].transform(X_eng)
        X_proc = best_pipeline.named_steps["preprocess"].transform(X_eng)
        X_sel = best_pipeline.named_steps["selector"].transform(X_proc)
        X_sel = best_pipeline.named_steps["to_dense"].transform(X_sel)

        # Get feature names if available
        try:
            feature_names = best_pipeline.named_steps["selector"].get_feature_names_out()
        except:
            try:
                feature_names = best_pipeline.named_steps["preprocess"].get_feature_names_out()
            except:
                feature_names = None

        explainer = shap.TreeExplainer(best_pipeline.named_steps["model"])
        shap_values = explainer.shap_values(X_sel)

        st.subheader("Decision Transparency: SHAP Plot")
        plt.figure(figsize=(10, 4))

        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value[0],
            shap_values[0],
            feature_names=feature_names,
            show=False,
            max_display=10
        )

        st.pyplot(plt.gcf())
        plt.clf()

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
                inp["label"],
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
        st.markdown(f"## Final Prediction: {res}")

        if res == "Fraud":
            st.error("⚠️ Prediction: Fraud")
        else:
            st.success("✅ Prediction: Legitimate")

        st.write("Inputs sent to model:")
        st.dataframe(input_row[MODEL_INFO["keys"]])

        display_explanation(input_row)

    else:
        st.error(res)
