import boto3
import os
import base64
import subprocess

# Load env variables from .env
with open('.env', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            key, val = line.strip().split('=', 1)
            os.environ[key] = val

os.environ['AWS_DEFAULT_REGION'] = 'ap-southeast-1'

print("1. Fetching ECR Login Password via boto3...")
client = boto3.client('ecr')
token = client.get_authorization_token()['authorizationData'][0]['authorizationToken']
password = base64.b64decode(token).decode('utf-8').split(':')[1]

print("2. Logging into Docker...")
subprocess.run(
    f"echo {password} | docker login --username AWS --password-stdin 324037324041.dkr.ecr.ap-southeast-1.amazonaws.com",
    shell=True, check=True
)

print("3. Building Docker Image...")
subprocess.run("docker build --platform linux/amd64 --provenance=false -t automodeler-lambda .", shell=True, check=True)

print("4. Tagging Docker Image...")
subprocess.run("docker tag automodeler-lambda:latest 324037324041.dkr.ecr.ap-southeast-1.amazonaws.com/automodeler-lambda:latest", shell=True, check=True)

print("5. Pushing Docker Image to AWS...")
subprocess.run("docker push 324037324041.dkr.ecr.ap-southeast-1.amazonaws.com/automodeler-lambda:latest", shell=True, check=True)

print("6. Updating AWS Lambda Function Code...")
try:
    lambda_client = boto3.client('lambda', region_name='ap-southeast-1')
    lambda_client.update_function_code(
        FunctionName='roy-lambda',
        ImageUri='324037324041.dkr.ecr.ap-southeast-1.amazonaws.com/automodeler-lambda:latest'
    )
    print("Successfully triggered Lambda update!")
except Exception as e:
    print(f"Warning: Failed to automatically update Lambda function code: {e}")
    print("You may need to update the image manually in the AWS Console.")

print("ALL DONE SUCCESS!")
