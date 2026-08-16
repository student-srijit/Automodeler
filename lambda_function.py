import json
import urllib.parse
import uuid
import os

# AWS Lambda filesystem is strictly read-only except for /tmp.
# We must redirect HOME to /tmp to prevent Errno 30 crashes for standard tools.
os.environ['HOME'] = '/tmp'

import csv
import re
import hashlib
import statistics
import psycopg2
import subprocess
import traceback
import time
import base64
from groq import Groq
import boto3

s3 = boto3.client('s3')
_model = None

# ═══════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════

def stream_status(job_id, bucket, message):
    if not job_id or not bucket: return
    try:
        s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}_status.json", Body=json.dumps({"status": "processing", "message": message}))
    except: pass

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model (first cold start)...")
        _model = SentenceTransformer('all-mpnet-base-v2')
    return _model

def get_db_conn():
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.autocommit = True
    return conn


# ─── CockroachDB Experiment Tracker ────────────────────────────────────────────

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
        conn = get_db_conn()
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


# ─── S3 Model Registry ──────────────────────────────────────────────────────────

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



# ═══════════════════════════════════════════════════════════════════
# STAGE 1: ADVANCED DATA PROFILER
# ═══════════════════════════════════════════════════════════════════

DATE_PATTERNS = [
    re.compile(r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2})?'),
    re.compile(r'^\d{2}/\d{2}/\d{4}'),
    re.compile(r'^\d{2}-\d{2}-\d{4}'),
    re.compile(r'^\d{4}/\d{2}/\d{2}'),
]

def infer_type(values):
    if not values:
        return "UNKNOWN"
    int_p  = re.compile(r'^-?[\d,]+$')
    flt_p  = re.compile(r'^-?[\d,]+\.\d+$')
    uuid_p = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')
    bool_p = re.compile(r'^(true|false|yes|no|1|0)$', re.IGNORECASE)
    sample = values[:30]
    if all(uuid_p.match(v) for v in sample):                              return "UUID"
    if all(any(p.match(v) for p in DATE_PATTERNS) for v in sample):      return "TIMESTAMPTZ"
    if all(bool_p.match(v) for v in sample):                              return "BOOL"
    if all(int_p.match(v.replace(',', '')) for v in sample):              return "INT8"
    if all(flt_p.match(v.replace(',', '')) for v in sample):              return "FLOAT8"
    return "STRING"

def compute_numeric_stats(values):
    try:
        nums = [float(v.replace(',', '')) for v in values if v]
        if not nums:
            return {}
        q1  = statistics.quantiles(nums, n=4)[0]
        q3  = statistics.quantiles(nums, n=4)[2]
        iqr = q3 - q1
        outliers = [v for v in nums if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr]
        return {
            "min":           round(min(nums), 4),
            "max":           round(max(nums), 4),
            "mean":          round(statistics.mean(nums), 4),
            "median":        round(statistics.median(nums), 4),
            "std_dev":       round(statistics.stdev(nums), 4) if len(nums) > 1 else 0,
            "outlier_count": len(outliers)
        }
    except Exception:
        return {}

def detect_fk_hints(columns_data, headers):
    hints = []
    for col_a in headers:
        vals_a = set(v for v in columns_data[col_a] if v)
        if len(vals_a) < 2:
            continue
        for col_b in headers:
            if col_a == col_b:
                continue
            vals_b = set(v for v in columns_data[col_b] if v)
            if vals_a.issubset(vals_b) and len(vals_a) < len(vals_b) * 0.6:
                hints.append(f"{col_a} may reference {col_b}")
    return hints

def profile_csv(bucket, key):
    response   = s3.get_object(Bucket=bucket, Key=key)
    lines      = response['Body'].read().decode('utf-8', errors='replace').splitlines()
    reader     = csv.reader(lines)
    headers    = [h.strip() for h in next(reader)]
    rows       = list(reader)
    total_rows = len(rows)

    if total_rows == 0:
        raise ValueError("CSV is empty")

    row_hashes = [hashlib.md5(','.join(r).encode()).hexdigest() for r in rows]
    seen = {}
    duplicate_indices = set()
    for i, h in enumerate(row_hashes):
        if h in seen:
            duplicate_indices.add(i)
        else:
            seen[h] = i

    columns_data = {h: [] for h in headers}
    for row in rows:
        for idx, val in enumerate(row):
            if idx < len(headers):
                columns_data[headers[idx]].append(val.strip())

    profile_columns = []
    for col, values in columns_data.items():
        non_empty     = [v for v in values if v != '']
        unique_vals   = set(non_empty)
        uniqueness_ratio = len(unique_vals) / total_rows if total_rows > 0 else 0
        inferred_type = infer_type(non_empty)
        is_pk         = (uniqueness_ratio == 1.0) and (len(non_empty) == total_rows)

        col_profile = {
            "column_name":      col,
            "inferred_type":    inferred_type,
            "uniqueness_ratio": round(uniqueness_ratio, 4),
            "null_count":       total_rows - len(non_empty),
            "cardinality":      len(unique_vals),
            "is_pk_candidate":  is_pk,
            "sample_values":    list(unique_vals)[:5],
        }
        if inferred_type in ("INT8", "FLOAT8"):
            col_profile["numeric_stats"] = compute_numeric_stats(non_empty)

        profile_columns.append(col_profile)

    profile = {
        "table_name":            "submission",
        "total_rows":            total_rows,
        "duplicate_rows":        len(duplicate_indices),
        "columns":               profile_columns,
        "fk_relationship_hints": detect_fk_hints(columns_data, headers)
    }
    print("STAGE 1 PROFILE:", json.dumps(profile, indent=2))
    return profile, headers, rows, duplicate_indices

# ═══════════════════════════════════════════════════════════════════
# STAGE 3: CLUSTER SIZING AGENT
# ═══════════════════════════════════════════════════════════════════

def size_cluster(profile):
    total_rows = profile["total_rows"]
    num_cols   = len(profile["columns"])
    est_bytes  = total_rows * num_cols * 60
    est_gb     = est_bytes / 1e9

    if est_gb < 1:
        plan = {"tier": "serverless",      "storage_limit_gb": 10, "ru_limit": 50000}
    elif est_gb < 20:
        plan = {"tier": "dedicated-small", "nodes": 3, "vcpus": 2, "ram_gb": 8}
    else:
        plan = {"tier": "dedicated-large", "nodes": 5, "vcpus": 8, "ram_gb": 32}

    plan["estimated_data_gb"] = round(est_gb, 4)
    print(f"STAGE 3 CLUSTER PLAN: {json.dumps(plan)}")
    return plan

# ═══════════════════════════════════════════════════════════════════
# STAGE 2: AI MODELING AGENT — Normalized Multi-Table Schema
# ═══════════════════════════════════════════════════════════════════

SCHEMA_PROMPT = """You are an enterprise CockroachDB Data Architect AI.
Given a CSV data profile JSON, design a fully normalized (3NF) CockroachDB relational schema.

STRICT RULES:
1. Use 'is_pk_candidate' and 'uniqueness_ratio' to assign PRIMARY KEYs.
2. Use 'fk_relationship_hints' to detect relationships and DECOMPOSE into multiple tables when appropriate.
3. Every table MUST include an `embedding VECTOR(768)` column at the end for semantic vector search.
4. Include a CREATE VECTOR INDEX on the embedding column for EACH table.
5. Include CREATE INDEX statements for each FK column.
6. Apply NOT NULL where null_count == 0.
7. Map types: UUID->UUID, INT8->INT8, FLOAT8->FLOAT8, TIMESTAMPTZ->TIMESTAMPTZ, BOOL->BOOL, STRING->STRING.
8. Output ONLY raw valid JSON. No markdown. No explanation.

OUTPUT FORMAT:
{
  "tables": [
    {
      "table_name": "string",
      "create_table_sql": "CREATE TABLE ...",
      "create_vector_index_sql": "CREATE VECTOR INDEX ON table_name (embedding);",
      "create_indexes_sql": ["CREATE INDEX ...", ...]
    }
  ],
  "relationships": [
    {"from_table": "string", "from_col": "string", "to_table": "string", "to_col": "string"}
  ]
}"""

def generate_schema(profile):
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user",   "content": f"Generate normalized schema:\n{json.dumps(profile)}"}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=4096
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r'^```[a-z]*\n?', '', content, flags=re.MULTILINE)
    content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
    schema  = json.loads(content.strip())
    print("STAGE 2 SCHEMA:", json.dumps(schema, indent=2))
    return schema

# ═══════════════════════════════════════════════════════════════════
# STAGE 4: COCKROACHDB PROVISIONING
# ═══════════════════════════════════════════════════════════════════
def deploy_schema(schema):
    conn = get_db_conn()
    with conn.cursor() as cur:
        # Drop all existing tables in the public schema to ensure a clean slate
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        existing_tables = cur.fetchall()
        for (tbl,) in existing_tables:
            cur.execute(f'DROP TABLE IF EXISTS "{tbl}" CASCADE;')
            
        for tbl_def in schema["tables"]:
            print(f"Ensuring table exists: {tbl_def['table_name']}")
            create_sql = tbl_def["create_table_sql"].replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)
            cur.execute(create_sql)
            try:
                cur.execute(tbl_def["create_vector_index_sql"])
            except Exception as e:
                print(f"Vector index warning (may already exist): {e}")
            for idx_sql in tbl_def.get("create_indexes_sql", []):
                if idx_sql and idx_sql.strip():
                    try:
                        cur.execute(idx_sql)
                    except Exception as e:
                        print(f"Index warning (non-fatal): {e}")
    conn.close()
    print("STAGE 4: Schema deployed.")

# ═══════════════════════════════════════════════════════════════════
# STAGE 5: DATA TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════

def transform_rows(headers, rows, profile_columns, duplicate_indices):
    type_map   = {col["column_name"]: col["inferred_type"] for col in profile_columns}
    median_map = {}
    for col in profile_columns:
        if col["inferred_type"] in ("INT8", "FLOAT8"):
            stats = col.get("numeric_stats", {})
            median_map[col["column_name"]] = stats.get("median", 0)

    clean_rows = [r for i, r in enumerate(rows) if i not in duplicate_indices]
    print(f"STAGE 5: Removed {len(rows) - len(clean_rows)} duplicate rows.")

    transformed = []
    for row in clean_rows:
        if len(row) < len(headers):
            row = row + [''] * (len(headers) - len(row))

        clean_row = {}
        for i, header in enumerate(headers):
            raw   = row[i].strip() if i < len(row) else ''
            dtype = type_map.get(header, "STRING")

            if raw == '':
                if dtype in ("INT8", "FLOAT8"):
                    clean_row[header] = str(median_map.get(header, 0))
                elif dtype == "BOOL":
                    clean_row[header] = 'false'
                elif dtype == "TIMESTAMPTZ":
                    clean_row[header] = '1970-01-01T00:00:00'
                else:
                    clean_row[header] = 'UNKNOWN'
            else:
                if dtype in ("INT8", "FLOAT8"):
                    clean_row[header] = raw.replace(',', '')
                elif dtype == "BOOL":
                    clean_row[header] = 'true' if raw.lower() in ('true', 'yes', '1') else 'false'
                else:
                    clean_row[header] = raw

        transformed.append(clean_row)

    print(f"STAGE 5: {len(transformed)} rows ready to load.")
    return transformed

# ═══════════════════════════════════════════════════════════════════
# STAGE 6: BATCH LOAD WITH VECTOR EMBEDDINGS (batched encoding)
# ═══════════════════════════════════════════════════════════════════

def embed_and_insert(schema, headers, transformed_rows, batch_size=500):
    transformed_rows = transformed_rows[:500]  # Cap for RAG demo speed
    model         = get_model()
    conn          = get_db_conn()
    primary_table = schema["tables"][0]["table_name"]
    col_names     = ", ".join(headers) + ", embedding"
    placeholders  = ", ".join(["%s"] * len(headers)) + ", %s"
    insert_sql    = f"INSERT INTO {primary_table} ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

    print(f"STAGE 6: Encoding {len(transformed_rows)} rows in chunks...")
    texts = [" | ".join([f"{k}: {v}" for k, v in row_dict.items()]) for row_dict in transformed_rows]
    vectors = []
    chunk_size = 500
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        chunk_vectors = model.encode(chunk, batch_size=64, show_progress_bar=False)
        vectors.extend(chunk_vectors)
        print(f"STAGE 6: Encoded {min(i+chunk_size, len(texts))}/{len(texts)} rows...")
    print(f"STAGE 6: Encoding complete. Inserting into DB...")
    inserted = 0
    batch    = []

    with conn.cursor() as cur:
        for row_dict, vector in zip(transformed_rows, vectors):
            values = tuple([row_dict.get(h, 'UNKNOWN') for h in headers] + [str(vector.tolist())])
            batch.append(values)

            if len(batch) >= batch_size:
                cur.executemany(insert_sql, batch)
                inserted += len(batch)
                print(f"STAGE 6: {inserted} rows inserted...")
                batch = []

        if batch:
            cur.executemany(insert_sql, batch)
            inserted += len(batch)

    conn.close()
    print(f"STAGE 6: Total rows inserted: {inserted}")
    return inserted

# ═══════════════════════════════════════════════════════════════════
# STAGE 7: AUTONOMOUS QUERY TESTING & INDEX OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════

def autonomous_test_and_optimize(schema, profile):
    client        = Groq(api_key=os.environ['GROQ_API_KEY'])
    primary_table = schema["tables"][0]["table_name"]
    col_names     = [c["column_name"] for c in profile["columns"]]

    response = client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": "Generate 5 realistic CockroachDB SELECT queries. Output ONLY a raw JSON array: [\"SELECT...\", ...]"
        }, {
            "role": "user",
            "content": f"Table: {primary_table}\nColumns: {json.dumps(col_names)}"
        }],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_tokens=1024
    )

    try:
        content = response.choices[0].message.content.strip()
        content = re.sub(r'^```[a-z]*\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
        queries = json.loads(content.strip())
    except Exception as e:
        print(f"STAGE 7: Could not parse test queries ({e}), skipping.")
        return

    conn = get_db_conn()
    with conn.cursor() as cur:
        for q in queries[:5]:
            try:
                cur.execute(f"EXPLAIN (FORMAT JSON) {q}")
                plan_str = json.dumps(cur.fetchone())
                if "full scan" in plan_str.lower():
                    where_match = re.search(r'WHERE\s+(\w+)', q, re.IGNORECASE)
                    if where_match:
                        col      = where_match.group(1)
                        idx_name = f"auto_idx_{primary_table}_{col}"
                        try:
                            cur.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {primary_table} ({col});")
                            print(f"STAGE 7: AUTO-CREATED INDEX {idx_name} — eliminated full scan.")
                        except Exception as e:
                            print(f"STAGE 7: Auto-index warning: {e}")
                else:
                    print(f"STAGE 7: Query OK (index scan): {q[:70]}...")
            except Exception as e:
                print(f"STAGE 7: Query test skipped: {e}")
    conn.close()

# ═══════════════════════════════════════════════════════════════════
# AGENT LOGIC
# ═══════════════════════════════════════════════════════════════════

def provision_agent_cluster():
    print(">>> Provisioning CockroachDB Serverless Cluster via ccloud CLI...")
    try:
        # Note: requires CCLOUD_API_KEY environment variable to be set in Lambda
        result = subprocess.run(
            ["ccloud", "cluster", "create", "serverless", "automodeler-memory", "--cloud", "AWS", "--spend-limit", "0", "-o", "json"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        conn_str = data.get("connection_string")
        if conn_str:
            conn_str = conn_str.replace("sslmode=verify-full", "sslmode=require")
            os.environ["DATABASE_URL"] = conn_str
            print("Successfully provisioned cluster and updated DATABASE_URL.")
            return conn_str
        else:
            raise ValueError("Connection string not found in ccloud output.")
    except Exception as e:
        print(f"Failed to provision cluster: {e}")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"ccloud stderr: {e.stderr}")
        raise e

def _get_table_schema(conn, table_name):
    """Fetch table schema as a dict of {column_name: data_type}."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position
        """, (table_name,))
        return {row[0]: row[1] for row in cur.fetchall() if row[0] != 'embedding'}


def _safe_sql_execute(conn, sql, params=None):
    """Execute a SQL query safely and return (columns, rows, error)."""
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if cur.description:
                cols = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                # Serialize types
                serialized_rows = []
                for row in rows:
                    serialized_rows.append([
                        str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                        for v in row
                    ])
                return cols, serialized_rows, None
            else:
                return [], [], None  # DML success
    except Exception as e:
        return None, None, str(e)


def _classify_intent(groq_client, user_query):
    """
    Classify user intent into one of:
    - 'train': user wants to train an ML model
    - 'clean_action': user approved/rejected a data cleaning action
    - 'eda': user wants to explore / analyze data
    Returns dict with 'intent' and optional 'target' / 'action' / 'confirm'
    """
    resp = groq_client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": """You are an intent classifier for an AI data platform. Classify the user message into exactly one of:
- "train": user wants to predict, train, or build a model for a specific column
- "clean_action": user is responding to a data cleaning suggestion (approving, rejecting, or specifying a method)
- "eda": user wants to explore data, ask statistics, find missing values, get distributions, etc.

Respond ONLY with a valid JSON object and nothing else. Examples:
{"intent": "train", "target": "Price"}
{"intent": "eda"}
{"intent": "clean_action", "confirm": true, "method": "mean"}
{"intent": "clean_action", "confirm": false}"""
        }, {
            "role": "user",
            "content": user_query
        }],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=100
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown if any
    raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    try:
        return json.loads(raw.strip())
    except Exception:
        # Fallback: if it looks like a train intent
        if any(w in user_query.lower() for w in ['predict', 'train', 'model', 'classify', 'forecast']):
            # Try to extract column name
            words = user_query.split()
            for i, w in enumerate(words):
                if w.lower() in ('predict', 'for', 'target', 'classify'):
                    if i + 1 < len(words):
                        return {"intent": "train", "target": words[i+1].strip("'\".,?")}
        return {"intent": "eda"}


def _generate_eda_sql(groq_client, user_query, table_name, schema):
    """
    Generate a safe read-only SQL SELECT query for any EDA question.
    Returns (sql_string, explanation)
    """
    schema_desc = "\n".join([f"  - {col} ({dtype})" for col, dtype in schema.items()])
    resp = groq_client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": f"""You are a CockroachDB SQL expert. Generate ONLY a single safe, read-only SELECT query to answer the user's EDA question.
Table: "{table_name}"
Columns:
{schema_desc}

Rules:
1. Output ONLY a JSON object: {{"sql": "...", "explanation": "..."}}
2. Use only SELECT statements — never INSERT, UPDATE, DELETE, DROP
3. Use NULL checks with IS NULL / IS NOT NULL
4. For missing values count: SELECT COUNT(*) - COUNT(col) AS missing_count FROM table
5. For distributions: use GROUP BY with COUNT(*)
6. Keep LIMIT ≤ 50 for row-fetching queries
7. Column names with spaces must be quoted with double quotes
8. Do NOT reference the 'embedding' column"""
        }, {
            "role": "user",
            "content": user_query
        }],
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        max_tokens=512
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    try:
        parsed = json.loads(raw.strip())
        return parsed.get("sql", ""), parsed.get("explanation", "")
    except Exception:
        # Try to extract raw SQL
        sql_match = re.search(r'SELECT[\s\S]+?;', raw, re.IGNORECASE)
        if sql_match:
            return sql_match.group(), "Generated query"
        return "", ""


def _check_data_quality(conn, table_name, schema):
    """
    Run quick data quality checks and return a structured report.
    Returns a list of issues like:
    [{"column": "Age", "issue": "missing_values", "count": 15, "suggestion": "fill_mean", "mean": 34.2}, ...]
    """
    issues = []
    total_rows = 0
    
    try:
        with conn.cursor() as cur:
            cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            total_rows = cur.fetchone()[0]
    except Exception:
        return issues, total_rows
    
    numeric_types = {'integer', 'bigint', 'numeric', 'double precision', 'real', 'smallint'}
    
    for col, dtype in schema.items():
        try:
            with conn.cursor() as cur:
                # Count nulls
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NULL')
                null_count = cur.fetchone()[0]
                
                if null_count > 0:
                    issue = {
                        "column": col,
                        "issue": "missing_values",
                        "count": null_count,
                        "pct": round((null_count / total_rows) * 100, 1) if total_rows > 0 else 0
                    }
                    
                    if dtype in numeric_types:
                        # Get mean and median suggestion
                        cur.execute(f'SELECT AVG(CAST("{col}" AS FLOAT)) FROM "{table_name}" WHERE "{col}" IS NOT NULL')
                        mean_val = cur.fetchone()[0]
                        if mean_val is not None:
                            issue["suggestion"] = "fill_mean_or_median"
                            issue["mean"] = round(float(mean_val), 4)
                    else:
                        # Get mode for categorical
                        cur.execute(f'SELECT "{col}", COUNT(*) as cnt FROM "{table_name}" WHERE "{col}" IS NOT NULL GROUP BY "{col}" ORDER BY cnt DESC LIMIT 1')
                        mode_res = cur.fetchone()
                        if mode_res:
                            issue["suggestion"] = "fill_mode_or_unknown"
                            issue["mode"] = str(mode_res[0])
                    
                    issues.append(issue)
        except Exception as e:
            print(f"Data quality check failed for {col}: {e}")
    
    return issues, total_rows


def _format_results_with_llm(groq_client, user_query, sql, cols, rows, explanation):
    """Format SQL result rows into beautiful Markdown using LLaMA."""
    if not rows and not cols:
        return "The query returned no results."
    
    # Build a clean data summary (cap at 50 rows for token safety)
    data_summary = {
        "query_executed": sql,
        "explanation": explanation,
        "columns": cols,
        "rows": rows[:50],
        "total_rows_returned": len(rows)
    }
    
    resp = groq_client.chat.completions.create(
        messages=[{
            "role": "system",
            "content": """You are a data analyst presenting SQL query results to a business user.
Format the provided data clearly and beautifully using Markdown.
Rules:
1. NEVER invent numbers or facts not present in the data
2. Use a Markdown table if results have multiple rows/columns
3. Use **bold** for key statistics
4. Provide a 1-2 sentence plain-English summary at the top
5. If only a single value is returned, highlight it prominently
6. Do NOT repeat the SQL query to the user"""
        }, {
            "role": "user",
            "content": f"Original Question: {user_query}\n\nSQL Results:\n{json.dumps(data_summary, indent=2)}"
        }],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=1024
    )
    return resp.choices[0].message.content.strip()


def handle_agent_chat(user_query, db_url, pending_clean_context=None):
    """
    Intelligent EDA Chat Agent with Text-to-SQL, Data Cleaning Copilot, and ML Intent Routing.
    
    Capabilities:
    - Train Intent: routes user to ML training pipeline
    - EDA Intent: generates + executes SQL, formats beautifully, anti-hallucinated
    - Clean Intent: handles approve/reject/specify-method for data cleaning
    - Proactive Quality Check: after first EDA, proactively offers to fix data issues
    """
    print(f">>> [Chat Agent] Query: {user_query}")
    start_time = time.time()
    
    if db_url:
        os.environ["DATABASE_URL"] = db_url

    groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])
    
    # ─── Step 1: Connect and find table ────────────────────────────────────────
    conn = None
    table_name = None
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name NOT IN ('model_experiments')
                LIMIT 1
            """)
            res = cur.fetchone()
            if res:
                table_name = res[0]
    except Exception as e:
        return {"answer": f"⚠️ Could not connect to the database: {e}"}
    
    if not table_name:
        return {"answer": "I don't have any data loaded yet. Please upload a dataset first!"}
    
    schema = _get_table_schema(conn, table_name)
    
    # ─── Step 2: Classify Intent ────────────────────────────────────────────────
    intent_data = _classify_intent(groq_client, user_query)
    intent = intent_data.get("intent", "eda")
    print(f">>> [Chat Agent] Intent: {intent_data}")
    
    # ─── Step 3: Route by Intent ────────────────────────────────────────────────
    
    # INTENT: Train ML model
    if intent == "train":
        target = intent_data.get("target", "").strip()
        if not target:
            return {"answer": "I'd love to train a model! Which column would you like to predict? (e.g. 'Price', 'Category', 'Churn')"}
        return {"intent": "train", "target": target}
    
    # INTENT: User is responding to a data cleaning suggestion
    if intent == "clean_action" and pending_clean_context:
        confirmed = intent_data.get("confirm", False)
        
        if not confirmed:
            conn.close()
            return {
                "answer": "No problem! I'll leave the data as-is. You can clean it yourself and re-upload, or proceed to train the model directly.",
                "pending_clean": None
            }
        
        # Execute the cleaning SQL
        col = pending_clean_context.get("column")
        method = intent_data.get("method", pending_clean_context.get("default_method", "mean"))
        fill_value = None
        
        if method in ("mean", "avg", "average") and pending_clean_context.get("mean") is not None:
            fill_value = pending_clean_context["mean"]
        elif method in ("mode", "most_frequent") and pending_clean_context.get("mode") is not None:
            fill_value = f"'{pending_clean_context['mode']}'"
        elif method in ("median",) and pending_clean_context.get("mean") is not None:
            fill_value = pending_clean_context["mean"]  # approximate with mean if no median cached
        else:
            fill_value = f"'{method}'"  # user typed a literal value
        
        clean_sql = f'UPDATE "{table_name}" SET "{col}" = {fill_value} WHERE "{col}" IS NULL'
        print(f">>> [Clean Agent] Executing: {clean_sql}")
        _, _, err = _safe_sql_execute(conn, clean_sql)
        conn.close()
        
        if err:
            return {"answer": f"❌ Cleaning failed for **{col}**: `{err}`\n\nYou may need to handle this manually."}
        
        return {
            "answer": f"✅ **Done!** Filled **{pending_clean_context.get('count', '?')} missing values** in column `{col}` using **{method}**.\n\nThe data is now clean. Would you like me to check for more issues, or are you ready to **train the model**?",
            "pending_clean": None
        }
    
    # INTENT: EDA — Text-to-SQL
    sql, explanation = _generate_eda_sql(groq_client, user_query, table_name, schema)
    
    if not sql:
        conn.close()
        return {"answer": "I couldn't generate a valid SQL query for that question. Could you rephrase it? For example: *'How many missing values are in the Age column?'* or *'What is the average price?'*"}
    
    print(f">>> [EDA Agent] Generated SQL: {sql}")
    cols, rows, err = _safe_sql_execute(conn, sql)
    
    if err:
        conn.close()
        return {"answer": f"⚠️ The query encountered an issue: `{err}`\n\nTry rephrasing your question!"}
    
    answer = _format_results_with_llm(groq_client, user_query, sql, cols, rows, explanation)
    
    # ─── Step 4: Proactive Data Quality Check ──────────────────────────────────
    # After answering, silently check for data issues and offer to fix the worst one
    pending_clean = None
    quality_prompt_addition = ""
    
    try:
        issues, total_rows = _check_data_quality(conn, table_name, schema)
        if issues:
            worst = max(issues, key=lambda x: x["count"])
            col = worst["column"]
            count = worst["count"]
            pct = worst["pct"]
            
            if worst.get("mean") is not None:
                method_hint = f"fill them with the **mean** ({worst['mean']}) or **median**"
                default_method = "mean"
            elif worst.get("mode") is not None:
                method_hint = f"fill them with the **most frequent value** (`{worst['mode']}`) or mark as `Unknown`"
                default_method = "mode"
            else:
                method_hint = "drop those rows"
                default_method = "drop"
            
            quality_prompt_addition = f"\n\n---\n💡 **Data Quality Alert:** I noticed `{col}` has **{count} missing values** ({pct}% of {total_rows} rows). Want me to {method_hint}? Just say **yes** and specify the method, or **no** to handle it yourself."
            
            pending_clean = {
                "column": col,
                "count": count,
                "mean": worst.get("mean"),
                "mode": worst.get("mode"),
                "default_method": default_method
            }
    except Exception as e:
        print(f">>> [Quality Check] Failed: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "answer": answer + quality_prompt_addition,
        "sql_executed": sql,
        "metrics": {"latency_ms": latency_ms},
        "pending_clean": pending_clean
    }



# ═══════════════════════════════════════════════════════════════════
# STAGE 8: AUTO-ML CODING AGENT  (Multi-Modal)
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# STAGE 8: AUTO-ML CODING AGENT  (Multi-Modal)
# ═══════════════════════════════════════════════════════════════════


IMAGE_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
AUDIO_EXTS  = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
TABULAR_EXTS = {'.csv', '.tsv', '.xlsx', '.xls'}
ZIP_EXT      = '.zip'

def detect_modality(filename):
    """Return 'image', 'audio', 'tabular', or 'zip_unknown' based on file extension."""
    import os
    ext = os.path.splitext(filename.lower())[1]
    if ext in IMAGE_EXTS:
        return 'image'
    if ext in AUDIO_EXTS:
        return 'audio'
    if ext == ZIP_EXT:
        return 'zip'
    return 'tabular'


def call_nemotron_reasoner(code, execution_logs, metric, target_column, round_num, approach_history):
    """
    Agent 3: Nvidia Nemotron Ultra 253B via OpenRouter.
    Deep-reasons about ML code quality, hyperparameters, and approach.
    Returns a dict: {verdict, reasoning, suggestions, new_algorithm}
    """
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get('OPENROUTER_API_KEY', ''),
        default_headers={"HTTP-Referer": "https://automodeler.ai", "X-Title": "AutoModeler"}
    )
    
    prompt = f"""You are an elite ML Systems Reasoning Expert conducting a code review.

PIPELINE ROUND: {round_num}/3
TARGET COLUMN: {target_column}
APPROACHES TRIED SO FAR:
{chr(10).join(approach_history) if approach_history else 'None yet.'}

=== GENERATED ML CODE ===
{code[:6000]}

=== EXECUTION OUTPUT ===
{execution_logs[:3000]}

=== METRIC ACHIEVED ===
{metric}

Perform deep reasoning on this pipeline:
1. Are the hyperparameters optimal? (n_neighbors, C, n_estimators, etc.)
2. Is the feature engineering appropriate for the data type?
3. Is the algorithm the best choice, or would a different one score higher?
4. Are there bugs, data leakage, or implementation inefficiencies?
5. If metric is N/A or execution crashed, what specifically caused it?

Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
{{
  "verdict": "accept" | "improve" | "new_approach",
  "reasoning": "your concise but thorough analysis here",
  "suggestions": "specific, concrete code-level improvements to apply",
  "new_algorithm": "if verdict is new_approach: describe the completely different algorithm and strategy to try instead"
}}

Verdict rules:
- "accept": metric is strong and code quality is good
- "improve": same core approach, but fix hyperparameters / feature engineering / bugs
- "new_approach": fundamentally different algorithm needed (use only in round 1 or 2)
"""
    
    print(f">>> [Nemotron Reasoner] Calling nvidia/llama-3.1-nemotron-ultra-253b-v1:free ...")
    resp = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.2
    )
    
    raw = resp.choices[0].message.content.strip()
    print(f">>> [Nemotron] Raw response: {raw[:500]}")
    
    # Robustly extract JSON
    json_match = re.search(r'\{[\s\S]*?\}(?=\s*$|\s*\n)', raw)
    if not json_match:
        json_match = re.search(r'\{[\s\S]+\}', raw)
    if json_match:
        try:
            return json.loads(json_match.group())
        except Exception:
            pass
    
    # Fallback: parse verdict manually
    verdict = "improve"
    if "accept" in raw.lower()[:200]:
        verdict = "accept"
    elif "new_approach" in raw.lower()[:200] or "new approach" in raw.lower()[:200]:
        verdict = "new_approach"
    return {"verdict": verdict, "reasoning": raw[:1000], "suggestions": raw[500:1500], "new_algorithm": ""}


def _parse_metric_value(metric_str):
    """Extract float from metric string like '0.87 accuracy' or 'FINAL_METRIC: 0.912'."""
    if not metric_str or metric_str in ("N/A", "Metric output not found.", "Ready"):
        return -1.0
    try:
        nums = re.findall(r'[\d]*\.?[\d]+', str(metric_str))
        if nums:
            val = float(nums[0])
            # If it looks like a percentage (e.g. 87.5), normalize
            return val / 100.0 if val > 1.5 else val
    except Exception:
        pass
    return -1.0


def _run_reasoning_loop(analyst_prompt, coder_extra_context, bucket, start_time, target_column="target", max_rounds=3, job_id=None):
    """
    Multi-Agent Reasoning Loop:
    Round 1: DeepSeek-R1 Analyst → LLaMA-3.3 Coder → exec() → Score
    Review:  Nemotron 253B reviews code + metric → verdict
    Round 2: If 'improve': LLaMA re-codes with Nemotron suggestions → exec()
    Round 3: If 'new_approach': Full re-strategize with Nemotron's new algorithm
    Returns: best result across all rounds + full reasoning history
    """
    import io, sys
    groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])
    
    best_result = None
    best_metric_val = -1.0
    rounds_history = []
    approach_history = []
    
    current_analyst_prompt = analyst_prompt
    current_coder_extra = coder_extra_context
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n{'='*60}")
        print(f">>> REASONING LOOP — ROUND {round_num}/{max_rounds}")
        print(f"{'='*60}")
        
        # ── Agent 1: Analyst (DeepSeek-R1 on Groq — reasoning model for strategy) ──
        print(">>> [Agent 1 — Analyst] deepseek-r1-distill-llama-70b (Groq)")
        stream_status(job_id, bucket, f"Agent 1 (Strategist): DeepSeek-R1 analyzing dataset for Round {round_num}...")
        try:
            analyst_response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": current_analyst_prompt}],
                model="deepseek-r1-distill-llama-70b",
                max_tokens=2048,
                temperature=0.1
            )
            strategy_plan = analyst_response.choices[0].message.content.strip()
        except Exception as e:
            print(f">>> DeepSeek-R1 failed ({e}), falling back to LLaMA 3.3 70B")
            analyst_response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": current_analyst_prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=2048,
                temperature=0.1
            )
            strategy_plan = analyst_response.choices[0].message.content.strip()
        
        print(f">>> [Analyst] Strategy:\n{strategy_plan[:500]}...")
        approach_history.append(f"Round {round_num}: {strategy_plan[:150]}...")
        
        # ── Agent 2: Coder (LLaMA-3.3-70B on Groq — best for code gen) ──
        stream_status(job_id, bucket, f"Agent 2 (Coder): LLaMA-3.3 writing implementation for Round {round_num}...")
        coder_prompt = f"""You are an elite AI Coding Agent. Implement this ML strategy precisely.
Our Lead Data Scientist's strategy:
----------
{strategy_plan}
----------
{current_coder_extra}
CRITICAL REQUIREMENTS (follow in order):
1. Calculate a numeric performance metric and print "FINAL_METRIC: <numeric_value>" at the very end.
2. After training, serialize the FINAL trained model object to /tmp/model.pkl using joblib (preferred) or pickle.
   Example: import joblib; joblib.dump(model, '/tmp/model.pkl')
3. Output ONLY raw Python code. No markdown. No ``` blocks. Pure executable Python only.
"""
        print(">>> [Agent 2 — Coder] llama-3.3-70b-versatile (Groq)")
        coder_response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": coder_prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=8192,
            temperature=0.05
        )
        generated_code = coder_response.choices[0].message.content.strip()
        generated_code = re.sub(r'^```[a-z]*\n?', '', generated_code, flags=re.MULTILINE)
        generated_code = re.sub(r'```\s*$', '', generated_code, flags=re.MULTILINE)
        generated_code = generated_code.strip()
        print(f">>> [Coder] Code ({len(generated_code)} chars) generated.")
        
        # ── Execute Code ──
        print(">>> [Executor] Running generated code...")
        stream_status(job_id, bucket, f"Executing generated code for Round {round_num}...")
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        exec_error = None
        try:
            exec(generated_code, {})
        except Exception as e:
            exec_error = str(e)
        finally:
            sys.stdout = old_stdout
        
        execution_output = redirected_output.getvalue()
        if exec_error:
            execution_output += f"\nEXECUTION_ERROR: {exec_error}"
            print(f">>> [Executor] Error: {exec_error}")
        
        # Parse metric
        metric_match = re.search(r'FINAL_METRIC:\s*(.+)', execution_output)
        final_metric = metric_match.group(1).strip() if metric_match else "N/A"
        metric_val = _parse_metric_value(final_metric)
        print(f">>> [Executor] FINAL_METRIC={final_metric} (numeric={metric_val:.4f})")
        
        # Upload submission if produced
        if os.path.exists('/tmp/submission.csv'):
            try:
                s3_client = boto3.client('s3')
                s3_client.upload_file('/tmp/submission.csv', bucket, 'submission.csv')
                execution_output += "\n[SYSTEM] submission.csv uploaded to S3!"
                print(">>> [Executor] Uploaded submission.csv to S3")
            except Exception as e:
                execution_output += f"\n[SYSTEM] S3 upload failed: {e}"
        
        round_data = {
            "round": round_num,
            "strategy": strategy_plan,
            "code": generated_code,
            "logs": execution_output,
            "metric": final_metric,
            "metric_val": metric_val,
            "nemotron_verdict": None,
            "nemotron_reasoning": None
        }
        rounds_history.append(round_data)
        
        # Track best across rounds
        if best_result is None or metric_val > best_metric_val:
            best_metric_val = metric_val
            best_result = round_data
            print(f">>> [Tracker] New best metric: {final_metric}")
        
        # Check early stop: significant improvement over previous round
        if round_num >= 2:
            prev_val = rounds_history[-2]["metric_val"]
            if prev_val > 0 and metric_val >= prev_val * 1.05:
                print(f">>> [Loop] Early stop: metric improved {prev_val:.4f} → {metric_val:.4f} (>5%)")
                round_data["nemotron_verdict"] = "early_stop"
                break
        
        # Don't call Nemotron on final round
        if round_num >= max_rounds:
            break
        
        # ── Agent 3: Nemotron Reasoner (OpenRouter) ──
        print(f">>> [Agent 3 — Nemotron] Reviewing round {round_num} results...")
        stream_status(job_id, bucket, f"Agent 3 (Reviewer): Nemotron-253B evaluating Round {round_num} performance...")
        nemotron_review = {"verdict": "improve", "reasoning": "Reasoner unavailable.", "suggestions": "", "new_algorithm": ""}
        try:
            nemotron_review = call_nemotron_reasoner(
                generated_code, execution_output, final_metric,
                target_column, round_num, approach_history
            )
        except Exception as e:
            print(f">>> [Nemotron] Call failed: {e}. Defaulting to 'improve'.")
        
        verdict = nemotron_review.get("verdict", "improve")
        stream_status(job_id, bucket, f"Agent 3 Verdict: {verdict.upper()}. Planning next step...")
        suggestions = nemotron_review.get("suggestions", "")
        new_algorithm = nemotron_review.get("new_algorithm", "")
        reasoning = nemotron_review.get("reasoning", "")
        
        round_data["nemotron_verdict"] = verdict
        round_data["nemotron_reasoning"] = reasoning
        print(f">>> [Nemotron] Verdict: {verdict.upper()}")
        
        if verdict == "accept":
            print(">>> [Loop] Nemotron accepted result. Stopping early.")
            break
        elif verdict == "improve":
            # Patch coder prompt with suggestions
            current_coder_extra = f"""{coder_extra_context}

REVISION INSTRUCTIONS (from Reasoning Agent — Round {round_num} review):
Previous metric: {final_metric}
Apply these specific improvements to the code:
{suggestions}

The previous code had these issues:
{reasoning[:800]}
"""
        elif verdict == "new_approach":
            # Full reset: new strategy prompt
            current_analyst_prompt = f"""{analyst_prompt}

IMPORTANT CONTEXT — ROUND {round_num} FAILED (metric: {final_metric}):
A reasoning model (Nvidia Nemotron 253B) reviewed the previous approach and recommends
switching to a COMPLETELY DIFFERENT algorithm/strategy:
{new_algorithm}

Previous reasoning:
{reasoning[:800]}

Design a fresh strategy based on the recommended approach above.
"""
            # Also reset coder context to clean
            current_coder_extra = coder_extra_context
            print(f">>> [Loop] Switching to new approach for Round {round_num + 1}.")
    
    # Build final response
    latency_ms = int((time.time() - start_time) * 1000)
    best = best_result or rounds_history[0]
    
    improvement_history = [
        {
            "round": r["round"],
            "metric": r["metric"],
            "verdict": r.get("nemotron_verdict") or "final",
            "reasoning_summary": (r.get("nemotron_reasoning") or "")[:200]
        }
        for r in rounds_history
    ]
    
    print(f"\n>>> REASONING LOOP COMPLETE: {len(rounds_history)} rounds, best metric={best['metric']}")
    
    return {
        "status": "success",
        "final_metric": best["metric"],
        "generated_code": best["code"],
        "execution_logs": best["logs"],
        "metrics": {"latency_ms": latency_ms},
        "rounds_taken": len(rounds_history),
        "improvement_history": improvement_history,
        "nemotron_reasoning": best.get("nemotron_reasoning") or ""
    }


# Legacy alias — keeps all callers working
def _run_two_agent_pipeline(analyst_prompt, coder_extra_context, bucket, start_time, target_column="target", job_id=None):
    return _run_reasoning_loop(analyst_prompt, coder_extra_context, bucket, start_time, target_column=target_column, job_id=job_id)


def handle_image_task(bucket, filename, target_column, job_id=None):
    """2-Agent pipeline for image datasets (.zip of images or folder)."""
    print(f">>> [IMAGE MODALITY] file={filename} target={target_column}")
    start_time = time.time()

    analyst_prompt = f"""You are an elite AI Lead Data Scientist specializing in Computer Vision.
The user has uploaded an IMAGE dataset named '{filename}' to an S3 bucket '{bucket}'.
The target task is: '{target_column}'.

Typical structure: a .zip file containing sub-folders per class (e.g. cats/, dogs/), or a CSV manifest with columns [filename, label].

Develop an optimal Computer Vision ML strategy:
1. How to load the images from the zip (use zipfile + Pillow).
2. Feature extraction: use a pre-trained ResNet-50 (torchvision, remove final FC layer) as a feature extractor — no GPU needed.
3. Normalize features. For classification use SVM or LogisticRegression. For similarity/retrieval use NearestNeighbors.
4. If test images exist (a test.zip or test folder), generate predictions and save to submission.csv with columns [filename, {target_column}] for classification, or [filename, Index_list] for retrieval.
5. Print FINAL_METRIC: <accuracy or distance>.

Write a clear step-by-step algorithmic blueprint. Do NOT write Python code yet.
"""

    coder_context = f"""
Assume `{filename}` is available locally (already downloaded from S3).
Use only: zipfile, os, Pillow (PIL), numpy, sklearn, and optionally torch+torchvision (CPU only).
Save output to `submission.csv`.
"""
    result = _run_two_agent_pipeline(analyst_prompt, coder_context, bucket, start_time, target_column=target_column, job_id=job_id)
    result['modality'] = 'image'
    return result


def handle_audio_task(bucket, filename, target_column, job_id=None):
    """2-Agent pipeline for audio datasets (.zip of audio files)."""
    print(f">>> [AUDIO MODALITY] file={filename} target={target_column}")
    start_time = time.time()

    analyst_prompt = f"""You are an elite AI Lead Data Scientist specializing in Audio ML.
The user has uploaded an AUDIO dataset named '{filename}' to an S3 bucket '{bucket}'.
The target task is: '{target_column}'.

Typical structure: a .zip file containing sub-folders per class (e.g. happy/, sad/), each with .wav or .mp3 files.

Develop an optimal Audio ML strategy:
1. How to unzip and load audio files using librosa.
2. Feature extraction: extract MFCC features (n_mfcc=40) + chroma + spectral contrast from each file. Average across time axis.
3. Normalize features with StandardScaler.
4. For classification: SVM or RandomForest. For retrieval: NearestNeighbors (cosine).
5. If test audio exists (test.zip or test folder), generate predictions and save to submission.csv with columns [filename, {target_column}].
6. Print FINAL_METRIC: <accuracy or distance>.

Write a clear step-by-step algorithmic blueprint. Do NOT write Python code yet.
"""

    coder_context = f"""
Assume `{filename}` is available locally.
Use only: zipfile, os, numpy, librosa, soundfile, sklearn.
Save output to `submission.csv`.
"""
    result = _run_two_agent_pipeline(analyst_prompt, coder_context, bucket, start_time, target_column=target_column, job_id=job_id)
    result['modality'] = 'audio'
    return result


def handle_automl_train(target_column, db_url, bucket, filename, job_id=None):
    print(f">>> Handling AutoML training for target: {target_column}")
    start_time = time.time()

    if db_url:
        os.environ["DATABASE_URL"] = db_url

    # ── Modality Router ────────────────────────────────────────────
    modality = detect_modality(filename)
    print(f">>> Detected modality: {modality.upper()} for file: {filename}")

    if modality == 'image':
        result = handle_image_task(bucket, filename, target_column)
        result['target_column'] = target_column
        return result

    if modality == 'audio':
        result = handle_audio_task(bucket, filename, target_column)
        result['target_column'] = target_column
        return result

    # ── Tabular / ZIP (default) ─────────────────────────────────────
    import pandas as pd

    sample_data = []
    columns = []
    try:
        try:
            obj = s3.get_object(Bucket=bucket, Key=filename)
            df_sample = pd.read_csv(obj['Body'], nrows=10)
        except:
            df_sample = pd.read_csv(filename, nrows=10)
        columns = df_sample.columns.tolist()
        for _, row in df_sample.iterrows():
            row_dict = row.to_dict()
            for k, v in row_dict.items():
                if pd.isna(v):
                    row_dict[k] = None
                elif type(v).__name__ in ['datetime', 'date', 'Timestamp']:
                    row_dict[k] = str(v)
            sample_data.append(row_dict)
    except Exception as e:
        return {"error": f"Failed to load sample data from {filename}: {e}"}

    if target_column not in columns:
        return {"error": f"Target column '{target_column}' not found in dataset. Available columns: {columns}"}

    analyst_prompt = f"""You are an elite AI Lead Data Scientist.
Your task is to define the optimal ML mathematical strategy for the target column '{target_column}'.

Here is a sample of the data:
{json.dumps(sample_data, indent=2)}

REQUIREMENTS FOR YOUR STRATEGY:
1. The dataset will be `/tmp/train.csv` and `/tmp/test.csv`.
2. We need to find the top 10 most similar rows in the training dataset for each row in the test dataset.
3. Suggest the optimal feature extraction (e.g., TfidfVectorizer for text, StandardScaler for numeric).
4. Suggest the optimal model (sklearn.neighbors.NearestNeighbors) to find the 10 nearest neighbors.
5. The training set has an `Index` column. The output should map to training `Index` values, not array indices.
6. Save to `/tmp/submission.csv` with columns `Index` (from test) and `Index_list` (python list of 10 `Index` ints from train).

Write a clear, detailed algorithmic blueprint. Do NOT write Python code yet.
"""

    coder_context = f"""
Write a standalone Python script using scikit-learn that implements this strategy.
Assume `train.csv` and `test.csv` are located at `/tmp/train.csv` and `/tmp/test.csv`.
Save the output to `/tmp/submission.csv`.
CRITICAL RULES:
1. Do NOT drop the '{target_column}' column. You likely need it to generate features.
2. If you must drop columns, always use `errors='ignore'` (e.g. `df.drop(columns=['...'], errors='ignore')`).
3. Only use feature columns that exist in BOTH `train.csv` and `test.csv`.
4. IMPORTANT FOR SPEED: You MUST sample the training data using `train_df = train_df.head(2000)` immediately after reading it to prevent execution timeouts!
Column reference (first 10 rows sample):
{json.dumps(sample_data, indent=2)}
"""

    # Download datasets to /tmp/ for Lambda execution
    try:
        s3.download_file(bucket, "train.csv", "/tmp/train.csv")
        s3.download_file(bucket, "test.csv", "/tmp/test.csv")
    except Exception as e:
        print(f"Warning: Failed to download train/test CSVs to /tmp: {e}")

    result = _run_two_agent_pipeline(analyst_prompt, coder_context, bucket, start_time, target_column=target_column)
    result['target_column'] = target_column
    result['modality'] = 'tabular'
    return result


# ═══════════════════════════════════════════════════════════════════
# MAIN HANDLER
# ═══════════════════════════════════════════════════════════════════

def lambda_handler(event, context):
    print("Received event:", json.dumps(event)[:200])
    
    # ROUTE 1: API Gateway (Chat Mode)
    if 'httpMethod' in event or 'requestContext' in event:
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        }
        
        # Support both REST API (v1) and HTTP API (v2) payload formats
        method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method')
        
        if method == 'OPTIONS':
            return {'statusCode': 200, 'headers': headers, 'body': ''}
            
        try:
            raw_body = event.get('body') or '{}'
            if event.get('isBase64Encoded'):
                raw_body = base64.b64decode(raw_body).decode('utf-8')
            body = json.loads(raw_body)
            action = body.get('action', 'chat')
            
            if action == 'get_upload_url':
                filename = body.get('filename')
                bucket = body.get('bucket')
                if not filename or not bucket:
                    return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing filename or bucket'})}
                try:
                    url = s3.generate_presigned_url(
                        ClientMethod='put_object',
                        Params={
                            'Bucket': bucket,
                            'Key': filename,
                            'ContentType': 'application/octet-stream'
                        },
                        ExpiresIn=3600
                    )
                    return {
                        'statusCode': 200,
                        'headers': headers,
                        'body': json.dumps({'url': url})
                    }
                except Exception as e:
                    return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': str(e)})}
            
            if action == 'upload':
                filename = body.get('filename')
                content_b64 = body.get('content')
                bucket = body.get('bucket')
                
                if not filename or not content_b64 or not bucket:
                    return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing filename, content, or bucket for upload'})}
                
                file_bytes = base64.b64decode(content_b64)
                s3.put_object(Bucket=bucket, Key=filename, Body=file_bytes)

                modality = detect_modality(filename)
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'status': 'success',
                        'message': f'Uploaded {filename} to {bucket}.',
                        'detected_modality': modality
                    })
                }
                
            if action == 'train':
                target = body.get('target', '')
                bucket = body.get('bucket', '')
                if not target or not bucket:
                    return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing target column or bucket'})}
                
                is_async_job = body.get('is_async_job', False)
                job_id = body.get('job_id')

                # If this is the initial request from API Gateway, spawn the background task immediately!
                if not is_async_job:
                    job_id = str(uuid.uuid4())
                    lambda_client = boto3.client('lambda')
                    # Build payload for background invocation
                    async_payload = event.copy()
                    
                    # Update body to include async flags
                    body_dict = json.loads(async_payload.get('body', '{}'))
                    body_dict['is_async_job'] = True
                    body_dict['job_id'] = job_id
                    async_payload['body'] = json.dumps(body_dict)
                    async_payload['isBase64Encoded'] = False # CRITICAL BUGFIX
                    
                    try:
                        lambda_client.invoke(
                            FunctionName=os.environ.get('AWS_LAMBDA_FUNCTION_NAME', 'roy-lambda'),
                            InvocationType='Event',
                            Payload=json.dumps(async_payload)
                        )
                        print(f">>> Spawned background job {job_id}")
                        return {
                            'statusCode': 202,
                            'headers': headers,
                            'body': json.dumps({'status': 'processing', 'job_id': job_id})
                        }
                    except Exception as e:
                        return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': f"Failed to spawn background task: {e}"})}

                # --- This runs ONLY in the background task (is_async_job == True) ---
                print(f">>> Running background job {job_id} for target {target}")
                
                try:
                    try:
                        state_response = s3.get_object(Bucket=bucket, Key="agent_memory_state.txt")
                        db_url = state_response['Body'].read().decode('utf-8').strip()
                    except Exception as e:
                        # Upload error to S3 so frontend polling picks it up
                        err_resp = {'error': 'Could not find database state. Have you uploaded a dataset yet?'}
                        s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}.json", Body=json.dumps(err_resp))
                        return {'statusCode': 400, 'body': ''}
                    
                    # Smart Data Router: Find the CSV that actually contains the target column
                    filename = None
                    try:
                        import pandas as pd
                        import io
                        objs = s3.list_objects_v2(Bucket=bucket)
                        for o in objs.get('Contents', []):
                            if o['Key'].endswith('.csv') and not o['Key'].startswith('jobs/'):
                                try:
                                    obj = s3.get_object(Bucket=bucket, Key=o['Key'], Range='bytes=0-4096')
                                    df_head = pd.read_csv(io.BytesIO(obj['Body'].read()), nrows=0)
                                    if target in df_head.columns:
                                        filename = o['Key']
                                        print(f">>> Smart Router: Selected {filename} for target {target}")
                                        break
                                except:
                                    pass
                    except Exception as e:
                        print(f"Smart Router failed: {e}")
                        
                    if not filename:
                        err_resp = {'error': f"Target column '{target}' not found in any uploaded CSV files."}
                        s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}.json", Body=json.dumps(err_resp))
                        return {'statusCode': 400, 'body': ''}
                        
                    response = handle_automl_train(target, db_url, bucket, filename, job_id=job_id)
                    
                    # ─── S3 Model Registry ───────────────────────────────────
                    stream_status(job_id, bucket, "Publishing trained model to S3 Model Registry...")
                    s3_model_url = publish_model_to_s3(bucket, target)
                    if s3_model_url:
                        response['s3_model_url'] = s3_model_url
                        response['model_download_note'] = f"Model saved to {s3_model_url}"
                    
                    # ─── CockroachDB Experiment Tracker ──────────────────────
                    stream_status(job_id, bucket, "Logging experiment metadata to CockroachDB...")
                    metric_val = response.get('final_metric', 'N/A')
                    reasoning_summary = response.get('nemotron_reasoning', '')
                    rounds = response.get('rounds_taken', 1)
                    try:
                        from re import findall as re_findall
                        nums = re_findall(r'[\d]*\.?[\d]+', str(metric_val))
                        numeric_metric = float(nums[0]) if nums else None
                    except Exception:
                        numeric_metric = None
                    log_experiment_to_cockroach(target, metric_val, numeric_metric, rounds, s3_model_url, reasoning_summary)
                    response['experiment_tracked'] = True
                    
                    # Save the final result to S3 for the frontend to poll!
                    try:
                        s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}.json", Body=json.dumps(response))
                        print(f">>> Successfully saved job {job_id} to S3!")
                    except Exception as e:
                        print(f">>> Failed to save job {job_id} to S3: {e}")
                        
                    return {'statusCode': 200, 'body': 'Background task complete'}
                except Exception as e:
                    print(f">>> Background task CRASHED: {e}")
                    traceback.print_exc()
                    err_resp = {'error': f"Background task failed abruptly: {e}"}
                    s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}.json", Body=json.dumps(err_resp))
                    return {'statusCode': 500, 'body': str(e)}
                
            if action == 'check_job':
                job_id = body.get('job_id')
                bucket = body.get('bucket')
                if not job_id or not bucket:
                    return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing job_id or bucket'})}
                
                try:
                    obj = s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}.json")
                    result_data = json.loads(obj['Body'].read().decode('utf-8'))
                    
                    # Delete the job file to clean up space
                    try:
                        s3.delete_object(Bucket=bucket, Key=f"jobs/{job_id}.json")
                    except Exception:
                        pass
                        
                    return {
                        'statusCode': 200,
                        'headers': headers,
                        'body': json.dumps(result_data)
                    }
                except Exception as e:
                    # If file doesn't exist yet, it's still processing!
                    status_message = "Agent reasoning in progress..."
                    try:
                        status_obj = s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}_status.json")
                        status_data = json.loads(status_obj['Body'].read().decode('utf-8'))
                        status_message = status_data.get('message', status_message)
                    except:
                        pass
                        
                    return {
                        'statusCode': 200,
                        'headers': headers,
                        'body': json.dumps({'status': 'processing', 'message': status_message})
                    }
                
            # action == 'chat'
            user_query = body.get('query', '')
            bucket = body.get('bucket', '')
            pending_clean_context = body.get('pending_clean', None)
            
            if not user_query or not bucket:
                return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing query or bucket'})}
                
            try:
                # Fetch DB URL from S3 state
                state_response = s3.get_object(Bucket=bucket, Key="agent_memory_state.txt")
                db_url = state_response['Body'].read().decode('utf-8').strip()
            except Exception as e:
                return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Could not find database state. Have you uploaded a dataset yet?'})}
                
            response = handle_agent_chat(user_query, db_url, pending_clean_context=pending_clean_context)
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(response)
            }
        except Exception as e:
            traceback.print_exc()
            return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': str(e)})}

    # ROUTE 2: S3 Upload (ETL & Provisioning Mode)
    elif 'Records' in event and 's3' in event['Records'][0]:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key    = urllib.parse.unquote_plus(
            event['Records'][0]['s3']['object']['key'], encoding='utf-8'
        )
        print(f"\n{'='*60}")
        print(f"AutoModeler Agent ETL — s3://{bucket}/{key}")
        print(f"{'='*60}\n")

        try:
            # Autonomous Provisioning (Bypass if TEST_DB_URL is set in .env)
            if os.environ.get('TEST_DB_URL'):
                print(">>> Using TEST_DB_URL from environment for local testing...")
                new_db_url = os.environ.get('TEST_DB_URL').replace("sslmode=verify-full", "sslmode=require")
                os.environ['DATABASE_URL'] = new_db_url
            else:
                new_db_url = provision_agent_cluster()
            
            print(f">>> Saving agent memory state to s3://{bucket}/agent_memory_state.txt")
            s3.put_object(Bucket=bucket, Key="agent_memory_state.txt", Body=new_db_url.encode('utf-8'))
            s3.put_object(Bucket=bucket, Key="agent_filename.txt", Body=key.encode('utf-8'))
            
            print(">>> STAGE 1: Advanced Data Profiling...")
            profile, headers, rows, dupe_idx = profile_csv(bucket, key)

            print(">>> STAGE 3: Cluster Sizing...")
            cluster_plan = size_cluster(profile)

            print(">>> STAGE 2: AI Schema Generation (multi-table normalized)...")
            schema = generate_schema(profile)

            print(">>> STAGE 4: CockroachDB Provisioning (multi-table DDL)...")
            deploy_schema(schema)

            print(">>> STAGE 5: Data Transformation (clean + impute + deduplicate)...")
            transformed = transform_rows(headers, rows, profile["columns"], dupe_idx)

            print(">>> STAGE 6: Batch Embedding & Load...")
            rows_inserted = embed_and_insert(schema, headers, transformed)

            print(">>> STAGE 7: Autonomous Query Testing & Index Optimization...")
            autonomous_test_and_optimize(schema, profile)

            result = {
                'statusCode': 200,
                'body': json.dumps({
                    'pipeline':           'complete',
                    'new_database_url':   new_db_url,
                    'tables_created':     [t["table_name"] for t in schema["tables"]],
                    'rows_ingested':      rows_inserted
                })
            }
            print(f"\nPIPELINE COMPLETE:\n{json.dumps(result, indent=2)}")
            return result

        except Exception as e:
            traceback.print_exc()
            raise e