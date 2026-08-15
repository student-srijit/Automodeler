from dotenv import load_dotenv

print("Loading environment variables from .env...")
load_dotenv()

import lambda_function

# Simulate the S3 trigger event
bucket_name = "automodeler-uploads-rimo-1234"
file_key = "train.csv"

event = {
    'Records': [{
        's3': {
            'bucket': {'name': bucket_name},
            'object': {'key': file_key}
        }
    }]
}

print(f"\n Manually Triggering ETL Pipeline for s3://{bucket_name}/{file_key}...")
print("Please wait while the AI generates the schema and calculates the embeddings...\n")

try:
    result = lambda_function.lambda_handler(event, None)
    print("\n✅ === PIPELINE FINISHED SUCCESSFULLY ===")
    print(result)
except Exception as e:
    print("\n❌ === PIPELINE FAILED ===")
    print(f"Error: {e}")
