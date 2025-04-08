import sagemaker
import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve
from sagemaker.inputs import TrainingInput

# Initialize session & role
sagemaker_session = PipelineSession()
role = "arn:aws:iam::419622399030:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"
s3_bucket = "s3://fraud-detectml"

# **Data Preprocessing Step**
preprocessing_processor = ScriptProcessor(
    image_uri="419622399030.dkr.ecr.us-east-1.amazonaws.com/sagemaker-preprocessing:latest",
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)
preprocessing_step = ProcessingStep(
    name="DataPreprocessing",
    processor=preprocessing_processor,
    inputs=[
        ProcessingInput(
            source=f"{s3_bucket}/dataset/creditcard.csv",
            destination="/opt/ml/processing/input/",
        )
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output/",
            destination=f"{s3_bucket}/processed-data/",
        )
    ],
    code="data_preprocessing.py",
)

# **Feature Store Step (Runs After Preprocessing)**
feature_store_processor = ScriptProcessor(
    image_uri="419622399030.dkr.ecr.us-east-1.amazonaws.com/sagemaker-preprocessing:latest",
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)
feature_store_step = ProcessingStep(
    name="FeatureStoreIngestion",
    processor=feature_store_processor,
    inputs=[
        ProcessingInput(
            source=f"{s3_bucket}/processed-data/train.csv",
            destination="/opt/ml/processing/input/",
        )
    ],
    code="feature_store-py",
    job_arguments=[
        "--feature-group-name", "fraud-detection-feature-store",
        "--role-arn", role
    ]
)
feature_store_step.add_depends_on([preprocessing_step])
# **Training Step (Runs After Feature Store)**
image_uri = retrieve("xgboost", region="us-east-1", version="1.5-1")
train_estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"{s3_bucket}/model-output/",
    sagemaker_session=sagemaker_session,
    entry_point="train.py",
    base_job_name="fraud-detection-job"
)
training_step = TrainingStep(
    name="ModelTraining",
    estimator=train_estimator,
    inputs={
        "train": TrainingInput(s3_data=f"{s3_bucket}/processed-data/train.csv", content_type="text/csv"),
        "test": TrainingInput(s3_data=f"{s3_bucket}/processed-data/test.csv", content_type="text/csv"),
    }
)
training_step.add_depends_on([feature_store_step])  # Ensure training runs AFTER Feature store

# Step 1: Model Registry
model_registry_processor = ScriptProcessor(
    image_uri="419622399030.dkr.ecr.us-east-1.amazonaws.com/sagemaker-preprocessing:latest",
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)

model_registry_step = ProcessingStep(
    name="ModelRegistry",
    processor=model_registry_processor,
    code="model_registry.py"
)

model_registry_step.add_depends_on([training_step])

# Step 1: Batch Inferencing
batch_inf_processor = ScriptProcessor(
    image_uri="419622399030.dkr.ecr.us-east-1.amazonaws.com/sagemaker-preprocessing:latest",
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)

batch_inf_step = ProcessingStep(
    name="BatchInferencing",
    processor=model_registry_processor,
    code="batch_inference.py"
)
# Step 2: Model Deployment
model_deploy_processor = ScriptProcessor(
    image_uri="419622399030.dkr.ecr.us-east-1.amazonaws.com/sagemaker-preprocessing:latest",
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    command=["python3"],
    sagemaker_session=sagemaker_session
)

model_deploy_step = ProcessingStep(
    name="ModelDeployment",
    processor=model_deploy_processor,
    code="deploy.py"
)
model_deploy_step.add_depends_on([batch_inf_step])
# **Create Pipeline**
pipeline = Pipeline(
    name="FraudDetectionPipeline",
    steps=[preprocessing_step, feature_store_step, training_step, model_registry_step, batch_inf_step, model_deploy_step]
)
#check for running pipelines
pipeline_executions = PipelineExecution.list(pipeline_name=pipeline_name, sagemaker_session=sagemaker_session)
running_executions = [
    execution for execution in pipeline_executions if execution.status == "Executing"
]
if running_executions:
    print(f"Pipeline '{pipeline_name}' is already running with execution ARN: {running_executions[0].arn}")
    print("Skipping new execution.")
else:
    # Start a pipeline execution
    print("Starting pipeline execution...")
    execution = pipeline.start()
    print(f"Pipeline execution started with ARN: {execution.arn}")