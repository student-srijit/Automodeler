import os
import json
import boto3
from core.vector_engine import VectorEngine

s3 = boto3.client('s3')

def stream_status(job_id, bucket, message):
    if not job_id or not bucket: return
    try:
        s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}_status.json", Body=json.dumps({"status": "processing", "message": message}))
    except: pass

def ensure_experiments_table(conn):
    """Create the model_experiments table if it doesn't exist yet."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_experiments (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                target_column STRING NOT NULL,
                final_metric STRING,
                metric_value FLOAT,
                rounds_taken INT,
                s3_model_url STRING,
                agent_reasoning JSONB,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """)
    print(">>> [Tracker] model_experiments table ready.")

def log_experiment_to_cockroach(target_column, final_metric, metric_value, rounds_taken, s3_model_url, reasoning_summary):
    """Log a completed ML experiment to CockroachDB for the Experiment Tracker."""
    try:
        conn = VectorEngine.get_db_conn()
        ensure_experiments_table(conn)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_experiments (target_column, final_metric, metric_value, rounds_taken, s3_model_url, agent_reasoning)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                target_column,
                final_metric,
                float(metric_value) if metric_value else None,
                rounds_taken,
                s3_model_url,
                json.dumps({"summary": reasoning_summary[:500] if reasoning_summary else ""})
            ))
        conn.close()
        print(f">>> [Tracker] Logged experiment for target='{target_column}' metric={final_metric}")
    except Exception as e:
        print(f">>> [Tracker] Failed to log experiment: {e}")

def publish_model_to_s3(bucket, target_column):
    """
    If the agent generated a model file at /tmp/model.pkl, upload it to the
    S3 model registry and return the S3 URI. Otherwise return None.
    """
    model_path = '/tmp/model.pkl'
    if not os.path.exists(model_path):
        print(">>> [Registry] No model.pkl found at /tmp — skipping registry upload.")
        return None
    try:
        registry_key = f"registry/{target_column}_model.pkl"
        s3.upload_file(model_path, bucket, registry_key)
        s3_uri = f"s3://{bucket}/{registry_key}"
        print(f">>> [Registry] Model published to {s3_uri}")
        return s3_uri
    except Exception as e:
        print(f">>> [Registry] Upload failed: {e}")
        return None

IMAGE_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
AUDIO_EXTS  = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
TABULAR_EXTS = {'.csv', '.tsv', '.xlsx', '.xls'}
ZIP_EXT      = '.zip'

def detect_modality(filename):
    """Return 'image', 'audio', 'tabular', or 'zip_unknown' based on file extension."""
    ext = os.path.splitext(filename.lower())[1]
    if ext in IMAGE_EXTS:
        return 'image'
    if ext in AUDIO_EXTS:
        return 'audio'
    if ext == ZIP_EXT:
        return 'zip'
    return 'tabular'
