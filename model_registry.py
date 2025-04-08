import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.image_uris import retrieve

# Define AWS region
region = "us-east-1"  # Change this to your Aws region

# Create a Boto session with the region
boto_session = boto3.Session(region_name=region)

# Define SageMaker Session
session = sagemaker.Session(boto_session=boto_session)
sagemaker_client = boto3.client("sagemaker", region_name=region)

ROLE_ARN = "arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"

# Get latest training job
response = sagemaker_client.list_training_jobs(SortBy="CreationTime", SortOrder="Descending")
training_job_name = response["TrainingJobSummaries"][0]["TrainingJobName"]
image_uri = retrieve("xgboost", region=region, version="1.5-1")

# Fetch model artifact
response = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
model_artifact_s3_path = response["ModelArtifacts"]["S3ModelArtifacts"]

# Register Model
model_name = f"fraud-detection-model-{training_job_name.split('-')[-1]}"
model = Model(
    image_uri=image_uri,
    model_data=model_artifact_s3_path,
    role=ROLE_ARN,
    sagemaker_session=session
)

# Define Model Package Group Name
model_package_group_name = "fraud-detection-model-group"

# Register Model into Model Registry with Inference Specification
model_package = model.register(
    model_package_group_name=model_package_group_name,
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    content_types=["text/csv"],
    response_types=["text/csv"],
)
print("Model Registered with Inference Specs!")

response = sagemaker_client.list_model_packages(
    ModelPackageGroupName=model_package_group_name
)
print("Registered Model Packages:", response)

sagemaker_client.update_model_package(
    ModelPackageArn=model_package.model_package_arn,
    ModelApprovalStatus="Approved"
)
print("Model Approved!")