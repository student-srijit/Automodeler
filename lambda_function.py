import json
import urllib.parse
import os

# AWS Lambda filesystem is strictly read-only except for /tmp.
# We must redirect all cache and config files to /tmp to prevent Errno 30 crashes.
os.environ['HOME'] = '/tmp'
os.environ['HF_HOME'] = '/tmp/huggingface'

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

def handle_agent_chat(user_query, db_url):
    print(f">>> Handling chat query: {user_query}")
    start_time = time.time()
    
    if db_url:
        os.environ["DATABASE_URL"] = db_url

    # 1. Embed user query
    model = get_model()
    query_vector = model.encode(user_query).tolist()
    
    # 2. Find table name dynamically
    conn = get_db_conn()
    table_name = None
    with conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' LIMIT 1;")
        res = cur.fetchone()
        if res:
            table_name = res[0]
            
    if not table_name:
        return {"answer": "I don't have any data loaded in my memory yet."}
        
    # 3. Vector search & Indexes
    context_rows = []
    active_indexes = []
    try:
        with conn.cursor() as cur:
            # Fetch actual query execution plan
            explain_sql = f"EXPLAIN SELECT * FROM {table_name} ORDER BY embedding <-> %s::vector LIMIT 5;"
            cur.execute(explain_sql, (str(query_vector),))
            plan = "Execution Plan: "
            for row in cur.fetchall():
                line = row[0]
                if 'vector search' in line:
                    plan += "Vector KNN Search -> "
                elif 'lookup join' in line:
                    plan += "Row Lookup Join -> "
                elif 'table:' in line:
                    idx = line.split('@')[-1].strip()
                    plan += f"Index({idx}) "
            active_indexes = [plan.strip()]
                
            # Vector search
            cur.execute(f"SELECT * FROM {table_name} ORDER BY embedding <-> %s::vector LIMIT 5;", (str(query_vector),))
            cols = [desc[0] for desc in cur.description]
            for row in cur.fetchall():
                row_dict = dict(zip(cols, row))
                if 'embedding' in row_dict:
                    del row_dict['embedding']
                context_rows.append(row_dict)
    except Exception as e:
        print(f"Vector search failed: {e}")
    finally:
        conn.close()
        
    # 4. Groq response
    client = Groq(api_key=os.environ['GROQ_API_KEY'])
    prompt = f"User Question: {user_query}\n\nContext from Database:\n{json.dumps(context_rows, indent=2)}\n\nAnswer the question based ONLY on the context provided. Format your response beautifully using Markdown. Use bullet points for listing products, bold text for product names, and short paragraphs to make it highly readable and organized for a normal user."
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    answer = response.choices[0].message.content.strip()
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "answer": answer, 
        "context_used": context_rows,
        "metrics": {
            "latency_ms": latency_ms,
            "active_indexes": active_indexes
        }
    }


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


def _run_two_agent_pipeline(analyst_prompt, coder_extra_context, bucket, start_time):
    """Shared 2-Agent pipeline: Analyst -> Coder -> exec() -> return result dict."""
    import io, sys
    groq_client = Groq(api_key=os.environ['GROQ_API_KEY'])

    # Agent 1: Analyst
    print(">>> Prompting Lead Analyst (LLaMA 3.3 70B)...")
    analyst_response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": analyst_prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    strategy_plan = analyst_response.choices[0].message.content.strip()
    print(">>> Strategy Developed:\n", strategy_plan)

    # Agent 2: Coder
    coder_prompt = f"""You are an elite AI Coding Agent.
Our Lead Data Scientist has provided the following architectural strategy:
----------
{strategy_plan}
----------
{coder_extra_context}
Calculate an appropriate metric if possible and print "FINAL_METRIC: <value>", otherwise print "FINAL_METRIC: Ready".
Output ONLY the raw Python code. Do not use markdown blocks. Just pure code.
"""
    print(">>> Prompting Senior Coder (LLaMA 3.3 70B)...")
    coder_response = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": coder_prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )
    generated_code = coder_response.choices[0].message.content.strip()
    generated_code = re.sub(r'^```[a-z]*\n?', '', generated_code, flags=re.MULTILINE)
    generated_code = re.sub(r'```\s*$', '', generated_code, flags=re.MULTILINE)
    print(">>> Code Generated:\n", generated_code)

    # Execute
    print("\n=== GENERATED CODE ===")
    print(generated_code)
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        exec(generated_code, {})
    except Exception as e:
        sys.stdout = old_stdout
        return {"error": f"Failed to execute generated code: {e}", "code": generated_code}
    sys.stdout = old_stdout
    execution_output = redirected_output.getvalue()
    print("\n=== EXECUTION LOGS ===")
    print(execution_output)

    metric_match = re.search(r'FINAL_METRIC:\s*(.+)', execution_output)
    final_metric = metric_match.group(1) if metric_match else "Metric output not found."
    latency_ms = int((time.time() - start_time) * 1000)
    return {
        "status": "success",
        "final_metric": final_metric,
        "generated_code": generated_code,
        "execution_logs": execution_output,
        "metrics": {"latency_ms": latency_ms}
    }


def handle_image_task(bucket, filename, target_column):
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
    result = _run_two_agent_pipeline(analyst_prompt, coder_context, bucket, start_time)
    result['modality'] = 'image'
    return result


def handle_audio_task(bucket, filename, target_column):
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
    result = _run_two_agent_pipeline(analyst_prompt, coder_context, bucket, start_time)
    result['modality'] = 'audio'
    return result


def handle_automl_train(target_column, db_url, bucket, filename):
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
1. The dataset will be `train.csv` and `test.csv` in the current directory.
2. We need to find the top 10 most similar rows in the training dataset for each row in the test dataset.
3. Suggest the optimal feature extraction (e.g., TfidfVectorizer for text, StandardScaler for numeric).
4. Suggest the optimal model (sklearn.neighbors.NearestNeighbors) to find the 10 nearest neighbors.
5. The training set has an `Index` column. The output should map to training `Index` values, not array indices.
6. Save to `submission.csv` with columns `Index` (from test) and `Index_list` (python list of 10 `Index` ints from train).

Write a clear, detailed algorithmic blueprint. Do NOT write Python code yet.
"""

    coder_context = f"""
Write a standalone Python script using scikit-learn that implements this strategy.
Assume `train.csv` and `test.csv` are in the current directory.
Column reference (first 10 rows sample):
{json.dumps(sample_data, indent=2)}
"""
    result = _run_two_agent_pipeline(analyst_prompt, coder_context, bucket, start_time)
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
            body = json.loads(raw_body)
            action = body.get('action', 'chat')
            
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
                try:
                    state_response = s3.get_object(Bucket=bucket, Key="agent_memory_state.txt")
                    db_url = state_response['Body'].read().decode('utf-8').strip()
                except Exception as e:
                    return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Could not find database state. Have you uploaded a dataset yet?'})}
                
                try:
                    file_response = s3.get_object(Bucket=bucket, Key="agent_filename.txt")
                    filename = file_response['Body'].read().decode('utf-8').strip()
                except:
                    filename = "train.csv"
                    
                response = handle_automl_train(target, db_url, bucket, filename)
                return {
                    'statusCode': 200 if 'error' not in response else 400,
                    'headers': headers,
                    'body': json.dumps(response)
                }
                
            # action == 'chat'
            user_query = body.get('query', '')
            bucket = body.get('bucket', '')
            
            if not user_query or not bucket:
                return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing query or bucket'})}
                
            try:
                # Fetch DB URL from S3 state
                state_response = s3.get_object(Bucket=bucket, Key="agent_memory_state.txt")
                db_url = state_response['Body'].read().decode('utf-8').strip()
            except Exception as e:
                return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Could not find database state. Have you uploaded a dataset yet?'})}
                
            response = handle_agent_chat(user_query, db_url)
            
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