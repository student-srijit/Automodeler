import json
import urllib.parse
import os
import csv
import re
import hashlib
import statistics
import psycopg2
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
# MAIN HANDLER
# ═══════════════════════════════════════════════════════════════════

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key    = urllib.parse.unquote_plus(
        event['Records'][0]['s3']['object']['key'], encoding='utf-8'
    )
    print(f"\n{'='*60}")
    print(f"AutoModeler Pipeline — s3://{bucket}/{key}")
    print(f"{'='*60}\n")

    try:
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
                'stages_executed':    [1, 2, 3, 4, 5, 6, 7],
                'tables_created':     [t["table_name"] for t in schema["tables"]],
                'rows_ingested':      rows_inserted,
                'duplicates_removed': profile["duplicate_rows"],
                'cluster_plan':       cluster_plan
            })
        }
        print(f"\nPIPELINE COMPLETE:\n{json.dumps(result, indent=2)}")
        return result

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        raise e