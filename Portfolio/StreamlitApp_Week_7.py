import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import pickle

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

import shap


# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '.'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features_pair

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
df_features = extract_features_pair()

MODEL_INFO = {
        "endpoint": aws_endpoint,
        "explainer": "explainer_pair.pkl",
        "pipeline": "finalized_pair_model.tar.gz",
        "keys": ["AMZN", "CRM"],
        "inputs": [{"name": k, "type": "number", "min": 0.0, "default": 0.0, "step": 10.0} for k in ["AMZN", "CRM"]]
}

@st.cache_resource
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]

    if not os.path.exists(filename):
        s3_client.download_file(
            Filename=filename,
            Bucket=bucket,
            Key=f"{key}/{os.path.basename(filename)}"
        )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)

@st.cache_resource
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(
            Filename=local_path,
            Bucket=bucket,
            Key=key
        )

    with open(local_path, "rb") as f:
        return pickle.load(f)

# Prediction Logic
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = int(pd.DataFrame(raw_pred).values[-1][0])
        mapping = {0: "SELL", 1: "HOLD", 2: "BUY"}
        return mapping.get(pred_val, pred_val), pred_val, 200
    except Exception as e:
        return f"Error: {str(e)}", None, 500

# Local Explainability
def display_explanation(input_df, session, aws_bucket, predicted_class):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join("explainer", explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    best_pipeline = load_pipeline(session, aws_bucket, "sklearn-pipeline-deployment")

    # Manual preprocessing to match notebook
    X_pair = best_pipeline.named_steps['pair_ind_5'].transform(input_df.copy())
    X_imp = best_pipeline.named_steps['imputer'].transform(X_pair)
    X_scaled = best_pipeline.named_steps['scaler'].transform(X_imp)
    X_selected = best_pipeline.named_steps['feature_selection'].transform(X_scaled)

    if hasattr(X_pair, 'columns'):
        feature_names_all = X_pair.columns.tolist()
    else:
        feature_names_all = [f"feature_{i}" for i in range(X_pair.shape[1])]

    selected_mask = best_pipeline.named_steps['feature_selection'].get_support()
    selected_feature_names = np.array(feature_names_all)[selected_mask]

    X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)

    shap_values = explainer(X_selected_df)

    st.subheader("🔍 Decision Transparency (SHAP)")

    fig = plt.figure(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, predicted_class], max_display=10, show=False)
    st.pyplot(fig, clear_figure=True)

    top_feature = selected_feature_names[0] if len(selected_feature_names) > 0 else "N/A"
    st.info(f"**Business Insight:** The model relied most heavily on the engineered pair features, with **{top_feature}** among the most important drivers shown here.")

# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'],
                value=inp['default'],
                step=inp['step']
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    # Prepare data
    base_df = df_features.copy()
    input_df = pd.concat(
        [base_df, pd.DataFrame([data_row], columns=base_df.columns)],
        ignore_index=True
    )

    res, pred_class, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df, session, aws_bucket, pred_class)
    else:
        st.error(res)
