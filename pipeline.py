import sagemaker
import boto3
import botocore
import time
from datetime import datetime

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.functions import Join
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve
from sagemaker.inputs import TrainingInput

# Initialize session
sagemaker_session = PipelineSession()
role = "arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"
s3_bucket = "s3://fraud-detectml"
region = sagemaker_session.boto_region_name
account_id = sagemaker_session.account_id()

preprocessing_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/sagemaker-preprocessing:latest"
xgboost_image_uri = retrieve("xgboost", region=region, version="1.5-1")

# Preprocessing Step
preprocessing_processor = ScriptProcessor(
    image_uri=preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)
preprocessing_step = ProcessingStep(
    name="DataPreprocessing",
    processor=preprocessing_processor,
    inputs=[ProcessingInput(
        source=f"{s3_bucket}/dataset/creditcard.csv",
        destination="/opt/ml/processing/input/")],
    outputs=[ProcessingOutput(
        output_name="processed_data",
        source="/opt/ml/processing/output/",
        destination=f"{s3_bucket}/processed-data/")],
    code="data_preprocessing.py",
)

# Feature Store Step
feature_store_processor = ScriptProcessor(
    image_uri=preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)
feature_store_step = ProcessingStep(
    name="FeatureStoreIngestion",
    processor=feature_store_processor,
    inputs=[ProcessingInput(
        source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri,
        destination="/opt/ml/processing/input/")],
    code="feature_store.py",
    job_arguments=[
        "--feature-group-name", "fraud-detection-feature-store",
        "--role-arn", role,
        "--input-data-path", "/opt/ml/processing/input/train.csv"
    ]
)
feature_store_step.add_depends_on([preprocessing_step])

# Training Step
train_estimator = Estimator(
    image_uri=xgboost_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"{s3_bucket}/model-output/",
    sagemaker_session=sagemaker_session,
    entry_point="train.py",
    base_job_name="fraud-detection-train-job",
    hyperparameters={
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
        "train": TrainingInput(
            s3_data=Join(on="/", values=[
                preprocessing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri,
                "train.csv"
            ]),
            content_type="text/csv"
        ),
        "test": TrainingInput(
            s3_data=Join(on="/", values=[
                preprocessing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri,
                "test.csv"
            ]),
            content_type="text/csv"
        )
    }
)
training_step.add_depends_on([feature_store_step])

# Model Registry Step
model_registry_processor = ScriptProcessor(
    image_uri=preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)
model_registry_step = ProcessingStep(
    name="ModelRegistry",
    processor=model_registry_processor,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=Join(on="/", values=[
                preprocessing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri,
                "test.csv"
            ]),
            destination="/opt/ml/processing/test_data"
        )
    ],
    code="model_registry.py",
    job_arguments=[
        "--model-path", "/opt/ml/processing/model/model.tar.gz",
        "--test-data-path", "/opt/ml/processing/test_data/test.csv",
        "--model-package-group-name", "FraudDetectionModelPackageGroup",
        "--model-approval-status", "PendingManualApproval"
    ]
)
model_registry_step.add_depends_on([training_step])

# Batch Inference Step
batch_inf_processor = ScriptProcessor(
    image_uri=preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    command=["python3"],
    sagemaker_session=sagemaker_session
)
batch_inf_step = ProcessingStep(
    name="BatchInferencing",
    processor=batch_inf_processor,
    inputs=[
        ProcessingInput(
            source=f"{s3_bucket}/batch-inference-input/",
            destination="/opt/ml/processing/batch_input"
        ),
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        )
    ],
    outputs=[ProcessingOutput(
        output_name="batch_output",
        source="/opt/ml/processing/batch_output",
        destination=f"{s3_bucket}/batch-inference-output/"
    )],
    code="batch_inference.py",
    job_arguments=[
        "--model-path", "/opt/ml/processing/model/model.tar.gz",
        "--input-data-path", "/opt/ml/processing/batch_input",
        "--output-data-path", "/opt/ml/processing/batch_output"
    ]
)
batch_inf_step.add_depends_on([model_registry_step])

# Model Deployment Step
model_deploy_processor = ScriptProcessor(
    image_uri=preprocessing_image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)
model_deploy_step = ProcessingStep(
    name="ModelDeployment",
    processor=model_deploy_processor,
    inputs=[ProcessingInput(
        source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        destination="/opt/ml/processing/model"
    )],
    code="deploy.py",
    job_arguments=[
        "--model-path", "/opt/ml/processing/model/model.tar.gz",
        "--endpoint-name", "fraud-detection-endpoint",
        "--instance-type", "ml.m5.large",
        "--initial-instance-count", "1"
    ]
)
model_deploy_step.add_depends_on([batch_inf_step])

# Define Pipeline
pipeline_name = "FraudDetectionPipeline"
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[],
    steps=[
        preprocessing_step,
        feature_store_step,
        training_step,
        model_registry_step,
        batch_inf_step,
        model_deploy_step
    ],
    sagemaker_session=sagemaker_session
)

# Main block to create and optionally start pipeline
if __name__ == "__main__":
    print(f"Running pipeline script for '{pipeline_name}'")
    try:
        upsert_response = pipeline.upsert(role_arn=role, description="Fraud Detection Pipeline CI/CD")
        print(f"Pipeline created or updated: {upsert_response['PipelineArn']}")

        sagemaker_client = boto3.client("sagemaker")
        running = sagemaker_client.list_pipeline_executions(
            PipelineName=pipeline_name,
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=5
        )["PipelineExecutionSummaries"]
        is_running = any(e["PipelineExecutionStatus"] in ["Executing", "Stopping"] for e in running)

        if not is_running:
            execution_name = f"exec-{time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())}"
            execution = pipeline.start(
                execution_display_name=execution_name,
                execution_description="Triggered by CodePipeline"
            )
            print(f"Started execution: {execution.arn}")
        else:
            print("Another execution is already running. Skipping start.")
    except Exception as e:
        print(f"ERROR: {e}")
        raise
