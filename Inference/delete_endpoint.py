import os
import boto3

REGION        = "eu-north-1"
ENDPOINT_FILE = os.path.join(os.path.dirname(__file__), "endpoint_name.txt")

with open(ENDPOINT_FILE, "r") as f:
    endpoint_name = f.read().strip()

if not endpoint_name:
    raise ValueError(f"No endpoint name found in {ENDPOINT_FILE}")

try:
    boto_session = boto3.session.Session(profile_name="HIBE", region_name=REGION)
    boto_session.client("sts").get_caller_identity()
except Exception:
    boto_session = boto3.session.Session(region_name=REGION)

sm_client = boto_session.client("sagemaker")

sm_client.delete_endpoint(EndpointName=endpoint_name)
print(f"Endpoint '{endpoint_name}' has been deleted.")
