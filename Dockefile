# Use Python 3.8 as base image
FROM python:3.8

# Set working directory
WORKDIR /app

# Install required Python libraries
RUN pip install pandas numpy scikit-learn boto sagemaker imbalanced-learn fsspec s3fs

# Set AWS default region
ENV AWS_REGION=us-east-1

# Keep the entrypoint open for SageMaker Pipeline
ENTRYPOINT ["python"]