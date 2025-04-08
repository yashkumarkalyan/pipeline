import boto3
import json
import sagemaker
from sagemaker.model import ModelPackage
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
import sys

# Define AWS region and s3 details
region = "us-east-1"
bucket_name = "fraud-detectml"
file_key = "processed-data/f1_score.json"

# Define F1 Score Threshold
THRESHOLD = 0.8
# Create a Boto session
boto_session = boto3.Session(region_name=region)
# Define SageMaker Session
session = sagemaker.Session(boto_session=boto_session)
sagemaker_client = boto3.client("sagemaker", region_name=region)
s3_client = boto3.client("s3")

ROLE_ARN ="arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"
model_package_group_name = "fraud-detection-model-group"
# Function to fetch F1 score from S3
def get_f1_score():
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content = response["Body"].read().decode("utf-8")
        return json.loads(content)
    except Exception as e:
        print(f"Error fetching F1 score from S3: {str(e)}")
        sys.exit(1)
# Fetch the F1 Score.
f1_score_data = get_f1_score()
if f1_score_data is None or "f1_score" not in f1_score_data:
    print("F1 score not found in JSON file or the key 'f1_score' is missing.")
    sys.exit(1)
f1_score = f1_score_data.get("f1_score")
print(f"Retrieved F1 Score: {f1_score}")
# Compare F1 Score and decide deployment
if f1_score < THRESHOLD:
    print(f"F1 Score ({f1_score}) is below the threshold ({THRESHOLD}). Not deploying the model.")
    sys.exit(0)
print("F1 score meets the threshold. Proceeding with deployment...")
# Get the latest approved model from the Model Registry
response = sagemaker_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name,
    SortBy="CreationTime",
    SortOrder="Descending",
    ModelApprovalStatus="Approved",
    MaxResults=1
)
if not response.get("ModelPackageSummaryList"):
    print("No approved models found in the registry. Exiting.")
    sys.exit(1)
# Extract latest approved model ARN
latest_model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
print(f"Using latest model package ARN: {latest_model_package_arn}")
# Deploy the latest model

model = ModelPackage(
    role=ROLE_ARN,
    model_package_arn=latest_model_package_arn,
    sagemaker_session=session
)
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    serializer=CSVSerializer(),
    deserializer=CSVDeserializer(),
    endpoint_name="fraud-detection-endpoint"
)
print("Model Deployed Successfully!")