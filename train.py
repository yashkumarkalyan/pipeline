import subprocess
import sys

# Install MLflow if not available
try:
    import mlflow
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow"])
    import mlflow  # Import again after installation

import argparse
import os
import pandas as pd
import xgboost as xgb
import json
import boto3
from sklearn.metrics import f1_score

# Parse Input Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
parser.add_argument("--test", type=str, default="/opt/ml/input/data/test")
parser.add_argument("--model_dir", type=str, default="/opt/ml/model")
args = parser.parse_args()

# Set MLflow Tracking URI
tracking_uri = "http://23.20.34.12:5000/"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("XGBoost-Fraud-Detection")

# Load the Training and Testing Data
train_data_path = os.path.join(args.train, "train.csv")
test_data_path = os.path.join(args.test, "test.csv")
print(f"Loading Training Data from: {train_data_path}")
print(f"Loading Testing Data from: {test_data_path}")
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# Split Features and Labels
X_train = train_data.drop(columns=["Class"])  # Features
y_train = train_data["Class"]  # Target Label
X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

# Train the Model with Hyperparameters (Log Parameters with MLflow)
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # Initialize the XGBoost model
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate Model Performance and Log Metrics
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"Model Training Completed! F1-Score: {f1}")

    # Log metrics to MLflow
    mlflow.log_metric("f1_score", f1)

    # Log the Model with MLflow
    mlflow.xgboost.log_model(model, "xgboost-model")

    # Save the Model in /opt/ml/model/ (Use bst format for XGBoost)
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.bst")
    model.save_model(model_path)  # Save in .bst format
    print(f"Model Saved Successfully at: {model_path}")

    # Log the saved model as an artifact to MLflow
    mlflow.log_artifact(model_path)

    # **Save F1-score separately in /opt/ml/output/**
    metrics_dir = "/opt/ml/output/"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "f1_score.json")

    f1_score_data = {"f1_score": f1}

    # Save F1-score as a JSON file
    with open(metrics_path, "w") as f:
        json.dump(f1_score_data, f)

    print(f"F1-score saved at: {metrics_path}")

    # **upload F1-score to S3 (with error handling)**
    s3_bucket = "fraud-detectml"
    s3_key = "processed-data/f1_score.json"

    try:
        s3_client = boto3.client("s3")
        s3_client.upload_file(metrics_path, s3_bucket, s3_key)
        print(f"F1-score uploaded to s3://{s3_bucket}/{s3_key}")
    except Exception as e:
        print(f"Failed to upload F1-score to S3: {e}")