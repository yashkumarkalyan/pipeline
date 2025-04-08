import sagemaker
import boto3
import botocore
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve
from sagemaker.inputs import TrainingInput
from datetime import datetime
import time # Using time for execution name uniqueness

print(f"SageMaker SDK version: {sagemaker.__version__}")
print(f"Boto3 version: {boto3.__version__}")

# Initialize session & role
sagemaker_session = PipelineSession() # Use PipelineSession for pipeline context
role = "arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"
s3_bucket = "s3://fraud-detectml" # Ensure this bucket exists
region = sagemaker_session.boto_region_name
account_id = sagemaker_session.account_id()

# Define ECR image URI (ensure this exists in your ECR)
preprocessing_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-preprocessing:latest"
xgboost_image_uri = retrieve("xgboost", region=region, version="1.5-1") # Or specify your preferred version

print(f"Using Role ARN: {role}")
print(f"Using S3 Bucket: {s3_bucket}")
print(f"Using Region: {region}")
print(f"Using Account ID: {account_id}")
print(f"Using Preprocessing Image: {preprocessing_image_uri}")
print(f"Using Training Image: {xgboost_image_uri}")

# **Data Preprocessing Step**
preprocessing_processor = ScriptProcessor(
    image_uri=preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session # Pass the session
)
preprocessing_step = ProcessingStep(
    name="DataPreprocessing",
    processor=preprocessing_processor,
    inputs=[
        ProcessingInput(
            source=f"{s3_bucket}/dataset/creditcard.csv", # Source data location
            destination="/opt/ml/processing/input/",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="processed_data", # Give outputs names
            source="/opt/ml/processing/output/",
            destination=f"{s3_bucket}/processed-data/",
        )
    ],
    code="data_preprocessing.py", # Script for this step
)

# **Feature Store Step (Runs After Preprocessing)**
feature_store_processor = ScriptProcessor(
    image_uri=preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session # Pass the session
)
feature_store_step = ProcessingStep(
    name="FeatureStoreIngestion",
    processor=feature_store_processor,
    inputs=[
        ProcessingInput(
            # Use the output from the previous step
            source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri,
            # Specifically target the train data output by preprocessing script
            # Adjust path if your preprocessing script outputs differently
            destination="/opt/ml/processing/input/", # Destination inside the container
        )
    ],
    code="feature_store.py", # Script for this step (corrected typo)
    job_arguments=[
        "--feature-group-name", "fraud-detection-feature-store", # Example argument
        "--role-arn", role,
        # Pass input data path dynamically if needed by script
        "--input-data-path", "/opt/ml/processing/input/train.csv" # Assuming preproc outputs train.csv
    ]
)
feature_store_step.add_depends_on([preprocessing_step])

# **Training Step (Runs After Feature Store)**
train_estimator = Estimator(
    image_uri=xgboost_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"{s3_bucket}/model-output/",
    sagemaker_session=sagemaker_session, # Pass the session
    entry_point="train.py", # Script for this step
    base_job_name="fraud-detection-train-job", # Base name for training job
    hyperparameters={ # Example hyperparameters
        'num_round': '100',
        'max_depth': '3',
        'eta': '0.2',
        'objective': 'binary:logistic'
     }
)
training_step = TrainingStep(
    name="ModelTraining",
    estimator=train_estimator,
    inputs={
        # Use the output from the preprocessing step
        "train": TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri + "train.csv",
            content_type="text/csv"
            ),
        "test": TrainingInput(
            s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri + "test.csv",
             content_type="text/csv"
             )
    }
)
training_step.add_depends_on([feature_store_step])  # Ensure training runs AFTER Feature store

# **Model Registry Step (Runs After Training)**
model_registry_processor = ScriptProcessor(
    image_uri=preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session # Pass the session
)
model_registry_step = ProcessingStep(
    name="ModelRegistry",
    processor=model_registry_processor,
    inputs=[
         ProcessingInput( # Pass model data from training step
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput( # Pass test data if needed for evaluation during registry
            source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri + "test.csv",
            destination="/opt/ml/processing/test_data"
        )
    ],
    code="model_registry.py", # Script for this step
    job_arguments=[ # Pass necessary info to the script
        "--model-path", "/opt/ml/processing/model/model.tar.gz",
        "--test-data-path", "/opt/ml/processing/test_data/test.csv",
        "--model-package-group-name", "FraudDetectionModelPackageGroup", # Example name
        "--model-approval-status", "PendingManualApproval" # Or Approved
    ]
)
model_registry_step.add_depends_on([training_step])

# **Batch Inferencing Step (Runs After Model Registry)**
# Note: Often Batch Inference might run independently or on a schedule,
# but here it's included sequentially after registry.
batch_inf_processor = ScriptProcessor( # Use a specific processor instance for this step
    image_uri=preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large", # May need more memory/compute for inference
    command=["python3"],
    sagemaker_session=sagemaker_session # Pass the session
)
batch_inf_step = ProcessingStep(
    name="BatchInferencing",
    processor=batch_inf_processor, # Corrected processor instance
    inputs=[
         # Input the data you want to run batch inference on
         ProcessingInput(
            source=f"{s3_bucket}/batch-inference-input/", # Example: new unlabeled data
            destination="/opt/ml/processing/batch_input"
        ),
         # Pass the registered model name or model data artifact
         # This depends on how batch_inference.py expects the model
         # Option 1: Pass model artifacts directly (if script handles loading)
         ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
         )
         # Option 2: Pass model package ARN from registry step (if script uses create_transform_job)
         # Needs model_registry.py to output this value, e.g., to an S3 file or use Properties
    ],
    outputs=[
         ProcessingOutput(
            output_name="batch_output",
            source="/opt/ml/processing/batch_output", # Where the script saves output
            destination=f"{s3_bucket}/batch-inference-output/"
         )
    ],
    code="batch_inference.py", # Script for this step
    job_arguments=[ # Pass necessary info
        "--model-path", "/opt/ml/processing/model/model.tar.gz", # If using Option 1
        "--input-data-path", "/opt/ml/processing/batch_input",
        "--output-data-path", "/opt/ml/processing/batch_output",
        # "--model-package-arn", model_registry_step.properties. # If using Option 2 & passing ARN
    ]
)
batch_inf_step.add_depends_on([model_registry_step]) # Runs after model is registered

# **Model Deployment Step (Runs After Batch Inferencing)**
# Note: Deployment might often be triggered by model approval status in registry,
# or could be conditional. Here it runs sequentially.
model_deploy_processor = ScriptProcessor(
    image_uri=preprocessing_image_uri, # Often uses a base image with boto3/sdk
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session # Pass the session
)
model_deploy_step = ProcessingStep(
    name="ModelDeployment",
    processor=model_deploy_processor,
     inputs=[ # Pass necessary info, e.g., model details from registry or training
         ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
         )
         # Or inputs related to the model package ARN from registry step
     ],
    code="deploy.py", # Script for this step
    job_arguments=[ # Arguments for your deployment script
        "--model-path", "/opt/ml/processing/model/model.tar.gz",
        "--endpoint-name", "fraud-detection-endpoint", # Name for the endpoint
        "--instance-type", "ml.m5.large", # Endpoint instance type
        "--initial-instance-count", "1" # Initial instance count
        # Potentially pass model package ARN from registry step if deploy.py uses that
    ]
)
model_deploy_step.add_depends_on([batch_inf_step]) # Example: Deploy after batch runs

# **Define the Pipeline Object**
pipeline_name = "FraudDetectionPipeline"
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[], # Add pipeline parameters if needed
    steps=[
        preprocessing_step,
        feature_store_step,
        training_step,
        model_registry_step,
        batch_inf_step,
        model_deploy_step # Add deployment step
        ],
    sagemaker_session=sagemaker_session # Pass the session
)


# Main execution block - This part runs when the script is executed directly
if __name__ == "__main__":
    print(f"Running pipeline script for '{pipeline_name}'")

    try:
        # Upsert the pipeline definition: Create or Update the pipeline in SageMaker
        print("Upserting pipeline definition...")
        upsert_response = pipeline.upsert(role_arn=role, description="Fraud Detection Pipeline CI/CD")
        print(f"Pipeline definition upserted successfully. ARN: {upsert_response['PipelineArn']}")

        # Check for currently running executions of THIS pipeline
        print(f"Checking for running executions of pipeline '{pipeline_name}'...")
        sagemaker_client = boto3.client("sagemaker")
        is_running = False
        try:
            # List recent executions, sorted by creation time descending
            running_executions = sagemaker_client.list_pipeline_executions(
                PipelineName=pipeline_name,
                SortBy='CreationTime',
                SortOrder='Descending',
                MaxResults=10 # Check a few recent ones
            )['PipelineExecutionSummaries']

            # Check if any of the recent executions are in a running or stopping state
            for execution in running_executions:
                status = execution['PipelineExecutionStatus']
                if status in ['Executing', 'Stopping']:
                    print(f"Found an active execution:")
                    print(f"  ARN: {execution['PipelineExecutionArn']}")
                    print(f"  Status: {status}")
                    print(f"  StartTime: {execution.get('StartTime', 'N/A')}")
                    is_running = True
                    break # Found one, no need to check further

        except sagemaker_client.exceptions.ResourceNotFoundException:
            print(f"Pipeline '{pipeline_name}' not found during status check (should not happen after upsert). Assuming not running.")
            is_running = False
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code == 'ValidationException' and 'does not exist' in str(e):
                 print(f"Pipeline '{pipeline_name}' not found during status check (ValidationException). Assuming not running.")
                 is_running = False
            else:
                 # Handle other potential API errors during check
                 print(f"Warning: Error checking pipeline status: {e}. Proceeding assuming not running.")
                 is_running = False # Default to not running if check fails unexpectedly

        # Start the pipeline execution only if no other execution is running
        if not is_running:
            print(f"No active executions found. Starting a new execution for pipeline '{pipeline_name}'...")
            try:
                # Create a unique name for the execution using a timestamp
                execution_display_name = f"exec-{time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())}"
                execution = pipeline.start(
                    execution_display_name=execution_display_name,
                    execution_description="Triggered by CodePipeline"
                    )
                print(f"Pipeline execution started successfully.")
                print(f"  Execution ARN: {execution.arn}")
                print(f"  Display Name: {execution_display_name}")
                # Optional: You could add execution.wait() here if the CodeBuild job
                # needs to wait for the pipeline to finish, but usually you don't.
            except Exception as start_error:
                print(f"ERROR: Failed to start pipeline execution: {start_error}")
                # Raise the error to potentially fail the CodeBuild job
                raise start_error
        else:
            print(f"Pipeline '{pipeline_name}' has an active execution. Skipping new execution start.")

        print("Pipeline script finished.")

    except Exception as e:
        print(f"ERROR: An error occurred during pipeline upsert or start preparation: {e}")
        # Raise the error to potentially fail the CodeBuild job
        raise e