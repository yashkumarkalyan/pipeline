import boto3
import sagemaker
import pandas as pd
import time
import sys

# AWS Configurations
region = "us-east-1"
s3_bucket = "fraud-detectml"
input_s3_path = "processed-data/test.csv"  # Original file (with Class column)
output_s3_path = "processed-data/test_inference.csv"  # File for inference (without Class)
batch_output_s3_path = "batch-inference-results/"
final_output_s3_path = "batch-inference-results/final_predictions.csv"  # Final results file
role_arn = "arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"
model_package_group_name = "fraud-detection-model-group"

# AWs Clients
boto_session = boto3.Session(region_name=region)
sagemaker_client = boto3.client("sagemaker", region_name=region)
s3_client = boto3.client("s3")
session = sagemaker.Session(boto_session=boto_session)

# Get Latest Approved Model from Model Registry
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

# • Load Test Data & Remove Target Column df -
df = pd.read_csv(f"s3://{s3_bucket}/{input_s3_path}")

# • Save the original features for drift detection
original_features = df.copy()

# Remove target column before inference
df.drop(columns=["Class"], inplace=True)

# Save & Upload Cleaned Data for Inference
df.to_csv("test_inference.csv", index=False, header=False)
s3_client.upload_file("test_inference.csv", s3_bucket, output_s3_path)
print(f"Cleaned test data uploaded: s3://{s3_bucket}/{output_s3_path}")

##create SageMaker Model from Model RegistEy
model_name = f"fraud-detection-model-{int(time.time())}"
response = sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={"ModelPackageName": latest_model_package_arn},
    ExecutionRoleArn=role_arn
)
print(f"Model created: {model_name}")

# • Start Batch Transform Job
batch_job_name = f"fraud-detection-batch-{int(time.time())}"
response = sagemaker_client.create_transform_job(
    TransformJobName=batch_job_name,
    ModelName=model_name,
    TransformInput={
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{s3_bucket}/{output_s3_path}"
            }
        },
        "ContentType": "text/csv",
        "SplitType": "Line"
    },
    TransformOutput={"S3OutputPath": f"s3://{s3_bucket}/{batch_output_s3_path}"},
    TransformResources={"InstanceType": "ml.m5.large", "InstanceCount": 1}
)
print(f"Batch inference started! Results will be saved in s3://{s3_bucket}/{batch_output_s3_path}")

# * Wait for batch job to complete
print("Waiting for batch job to complete... (Check in AWS Console)")
time.sleep(150)

# * Download Batch Inference Results
batch_results_file = "batch_inference_results.csv"
# Find the actual output file in s3
s3_objects = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=batch_output_s3_path)
output_file_key = None
for obj in s3_objects.get("Contents", []):
    if obj["Key"].endswith(".out"):
        output_file_key = obj["Key"]
        break

# ADD PRINT STATEMENTS IF RESULTS NOT FOUND IN S3
if output_file_key is None:
    print(f"Error: No batch inference output file found in s3://{s3_bucket}/{batch_output_s3_path}")
    print("Please check the AWS Console for the status of the Batch Transform Job.")
    sys.exit(1)

s3_client.download_file(s3_bucket, output_file_key, batch_results_file)

# Convert Probabilities to Class Labels
df_preds = pd.read_csv(batch_results_file, header=None)
df_preds["Predicted_Probability"] = df_preds[0]
df_preds["Predicted_Class"] = (df_preds["Predicted_Probability"] >= 0.5).astype(int)

# • Merge Original Features with Predictions
final_df = pd.concat([original_features, df_preds[["Predicted_Probability", "Predicted_Class"]]], axis=1)

# * * Save Final Processed Results
final_df.to_csv("final_predictions.csv", index=False)

# Upload Final Predictions to S3
s3_client.upload_file("final_predictions.csv", s3_bucket, final_output_s3_path)
print(f"Processed results saved in S3: s3://{s3_bucket}/{final_output_s3_path}")