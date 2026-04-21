import os
import time
import tarfile
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

# ===== CONFIG =====
AWS_PROFILE     = "HIBE"
REGION          = "eu-north-1"
ROLE_ARN        = "arn:aws:iam::600889066998:role/SageMaker-ExecutionRole"
S3_MODEL_SRC    = "s3://hiba-slr-filer/best_model_2_1774971874.pth"
S3_MODEL_DST    = "s3://hiba-slr-filer/inference/model/model.tar.gz"
S3_BUCKET       = "hiba-slr-filer"
S3_KEY_DST      = "inference/model/model.tar.gz"
MODEL_FILENAME  = "best_model_2_1774971874.pth"
ENDPOINT_FILE   = os.path.join(os.path.dirname(__file__), "endpoint_name.txt")

INFERENCE_DIR   = os.path.dirname(__file__)
TRAINING_DIR    = os.path.join(os.path.dirname(__file__), "..", "training")
CNN_LSTM_SRC    = os.path.join(TRAINING_DIR, "cnn_lstm.py")

# ===== AWS SESSION =====
boto_session = boto3.session.Session(profile_name=AWS_PROFILE, region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)
s3_client = boto_session.client("s3")

# ===== STEP 1: Download model from S3 =====
local_model_path = os.path.join(INFERENCE_DIR, MODEL_FILENAME)
print(f"Downloading model from {S3_MODEL_SRC} ...")
s3_client.download_file(S3_BUCKET, MODEL_FILENAME, local_model_path)
print("Download complete.")

# ===== STEP 2: Package model.tar.gz =====
tar_path = os.path.join(INFERENCE_DIR, "model.tar.gz")
sm_inference_src = os.path.join(INFERENCE_DIR, "sm_inference.py")

print(f"Creating {tar_path} ...")
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(local_model_path, arcname=MODEL_FILENAME)
    tar.add(sm_inference_src, arcname="sm_inference.py")
    tar.add(CNN_LSTM_SRC, arcname="cnn_lstm.py")
print("Packaging complete.")

# ===== STEP 3: Upload model.tar.gz to S3 =====
print(f"Uploading to {S3_MODEL_DST} ...")
s3_client.upload_file(tar_path, S3_BUCKET, S3_KEY_DST)
print("Upload complete.")

# ===== STEP 4: Deploy to SageMaker =====
pytorch_model = PyTorchModel(
    model_data=S3_MODEL_DST,
    role=ROLE_ARN,
    framework_version="2.0.0",
    py_version="py310",
    entry_point="sm_inference.py",
    sagemaker_session=sagemaker_session,
)

print("Deploying to SageMaker (this measures cold-start time for RQ2)...")
deploy_start = time.time()

predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
)

deploy_seconds = time.time() - deploy_start
endpoint_name = predictor.endpoint_name

# ===== STEP 5: Save endpoint name =====
with open(ENDPOINT_FILE, "w") as f:
    f.write(endpoint_name)

print(f"\nDeployment complete.")
print(f"Endpoint name : {endpoint_name}")
print(f"Deploy time   : {deploy_seconds:.1f} seconds  (RQ2 cold-start time)")
print(f"Endpoint name written to: {ENDPOINT_FILE}")

# Cleanup local files
os.remove(local_model_path)
os.remove(tar_path)
