import boto3
import pandas as pd
import sagemaker
import uuid
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import FeatureDefinition, FeatureTypeEnum
from sagemaker import get_execution_role

# Set AWS Region
REGION = "us-east-1"  # Change to your Aws region
boto3.setup_default_session(region_name=REGION)  # Set default region

# Initialize SageMaker session with a specific region
boto_session = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"

# Define Feature Group Name & S3 Paths
FEATURE_GROUP_NAME = "fraud-detection-feature-store"
BUCKET_NAME = "fraud-detectml1"
S3_TRAIN_DATA = f"s3://{BUCKET_NAME}/processed-data/train.csv"
OFFLINE_STORE_PATH = f"s3://{BUCKET_NAME}/feature-store/"

# Load Data from S3
print(" Downloading processed train.csv from S3...")
s3 = boto3.client("s3", region_name=REGION)
s3.download_file(BUCKET_NAME, "processed-data/train.csv", "train.csv")

# Read Data
train_df = pd.read_csv("train.csv").head(5)
train_df["TransactionID"] = [str(uuid.uuid4()) for _ in range(len(train_df))]

# Define Feature Group Schema
feature_group = FeatureGroup(
    name=FEATURE_GROUP_NAME,
    feature_definitions=[
        FeatureDefinition(feature_name="TransactionID", feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name="Time", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="Amount", feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name="Class", feature_type=FeatureTypeEnum.INTEGRAL),
    ] + [
        FeatureDefinition(feature_name=f"V{i}", feature_type=FeatureTypeEnum.FRACTIONAL) for i in range(1, 29)
    ],
    sagemaker_session=sagemaker_session
)

# Create Feature Group (if not exists)
try:
    print(" Creating Feature Group...")
    feature_group.create(
        record_identifier_name="TransactionID",
        event_time_feature_name="Time",
        role_arn=role,
        enable_online_store=True,
        s3_uri=OFFLINE_STORE_PATH
    )
    print(" Feature Group Created Successfully!")
except Exception as e:
    print(f"Error creating Feature Group: {e}")

# Ingest Data into Feature Store
data_to_ingest = train_df[['TransactionID', 'Time', 'Amount', 'Class'] + [f"V{i}" for i in range(1, 29)]]
print(" Ingesting data into Feature Store...")
feature_group.ingest(data_frame=data_to_ingest, max_workers=3, wait=True)
print(" 5 Rows Ingested into Feature Store!")