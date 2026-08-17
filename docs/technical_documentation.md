# AutoModeler Technical Documentation

## 1. System Overview

AutoModeler is a serverless data engineering and machine learning pipeline that autonomously processes raw CSV files uploaded to Amazon S3. Triggered by S3 events, the system extracts data profiles, generates visual Exploratory Data Analysis (EDA) artifacts, provisions CockroachDB Serverless clusters dynamically, generates a database schema using an LLM, embeds data rows for semantic search, and automatically indexes the database. It also includes an AutoML agent that writes and executes Python code iteratively to train predictive models, and an intelligent chat agent to query the resulting database via Retrieval-Augmented Generation (RAG).

## 2. Architecture

The system relies on AWS Lambda as the primary compute environment, orchestrated by `pipeline.py`. Data is stored in Amazon S3 (raw files, generated plots, and state files) and CockroachDB (structured data and vectors). LLMs are utilized for schema generation, query optimization, machine learning reasoning loops, and chat interfaces.

<image src='../assets/pipeline.png' width=500 style="border-radius: 12px">

## 3. Pipeline

The actual execution flow triggered by an S3 upload, managed sequentially by `AutoModelerPipeline.run_s3_etl` in `pipeline.py`:

- **Stage 0: Cluster Provisioning & State Management**

  - **Component:** `ClusterProvisioner`

  - **Action:** If `TEST_DB_URL` is not set, it executes the `ccloud` CLI to provision a new serverless cluster. Saves the resulting connection string to `agent_memory_state.txt` in S3.

- **Stage 1: Advanced Data Profiling**

  - **Component:** `DataProfiler`

  - **Action:** Receives the CSV from S3, samples data to infer PostgreSQL types (UUID, TIMESTAMPTZ, INT8, etc.), and computes statistical boundaries (quantiles, IQR). Produces a JSON data profile and the raw parsed rows.

- **Stage 1.5: Visual EDA Generation**

  - **Component:** `VisualProfiler`

  - **Action:** Receives parsed rows and headers. Computes correlation matrices and distributions using pandas. Produces PNG images and saves them to S3 under `eda-output/{filename}/`.
  
- **Stage 2: AI Schema Generation**

  - **Component:** `SchemaArchitect`

  - **Action:** Receives the JSON data profile. Prompts an LLM to generate exactly one denormalized `CREATE TABLE` statement and a `CREATE VECTOR INDEX` statement. Ensures all table and column identifiers are double-quoted. Produces a JSON schema object.

- **Stage 3 & 4: Schema Deployment**

  - **Component:** `VectorEngine`

  - **Action:** Receives the generated schema. Drops existing tables matching the name and executes the generated DDL statements on the CockroachDB cluster.

- **Stage 5: Data Transformation**

  - **Component:** `DataTransformer`

  - **Action:** Receives the raw rows and profile. Cleans boolean variations, handles missing values (using median imputation for numerics), and deduplicates rows using MD5 hashing. Produces a list of clean dictionaries.

- **Stage 6: Batch Embedding & Load**

  - **Component:** `VectorEngine`

  - **Action:** Receives transformed rows. Uses `all-mpnet-base-v2` locally in chunks of 500 to generate 768-dimensional embeddings for a string representation of each row. Constructs dynamic `INSERT INTO` queries with double-quoted identifiers and bulk inserts data and vectors into CockroachDB using `psycopg2.executemany`.

- **Stage 7: Autonomous Query Testing & Index Optimization**

  - **Component:** `QueryOptimizer`

  - **Action:** Receives the schema and profile. Asks an LLM to generate sample analytical queries. Executes `EXPLAIN` on these queries against the database, analyzes the output for "full table scans", and if found, asks the LLM to recommend and execute `CREATE INDEX` statements.

## 4. Components

### a. DataProfiler

- **File:** `core/profiler.py`

- **Purpose:** Analyzes CSV datasets to infer types and calculate statistics.

- **Inputs:** S3 bucket, object key, s3 boto3 client.

- **Outputs:** JSON profile dictionary, list of headers, list of row dictionaries, list of duplicate indices.

- **Key methods:** `infer_type(values)`, `compute_numeric_stats(values)`, `profile_csv(bucket, key, s3_client)`

### b. VisualProfiler

- **File:** `core/visual_profiler.py`

- **Purpose:** Generates EDA plots via headless matplotlib.

- **Inputs:** S3 bucket, object key, s3 client, headers, row data.

- **Outputs:** Uploads PNG files to S3; returns a list of uploaded object keys.

- **Key methods:** `generate_and_upload_eda(...)`

### c. AutoMLAgent

- **File:** `core/auto_ml.py`

- **Purpose:** Automatically writes, executes, and iterates on scikit-learn Python code to train ML models.

- **Inputs:** Target column, DB URL, S3 bucket, filename.

- **Outputs:** Model performance metrics, reasoning summaries.

- **Key methods:** `handle_automl_train(...)`, `run_reasoning_loop(...)`, `call_nemotron_reasoner(...)`

### d. ClusterProvisioner

- **File:** `core/provisioner.py`

- **Purpose:** Calculates storage needs and autonomously creates CockroachDB clusters.

- **Inputs:** S3 file profile.

- **Outputs:** CockroachDB connection string.

- **Key methods:** `size_cluster(profile)`, `provision_agent_cluster()`

### e. SchemaArchitect

- **File:** `core/architect.py`

- **Purpose:** Uses an LLM to generate the CockroachDB SQL Schema based on the data profile.

- **Inputs:** Data profile JSON.

- **Outputs:** JSON object containing `table_name` and `create_table_sql`.

- **Key methods:** `generate_schema(profile)`

### f. DataTransformer

- **File:** `core/transformer.py`

- **Purpose:** Cleans and imputes missing data based on the expected schema.

- **Inputs:** Headers, rows, column profiles, duplicate indices.

- **Outputs:** Cleaned list of row dictionaries.

- **Key methods:** `transform_rows(...)`

### g. VectorEngine

- **File:** `core/vector_engine.py`

- **Purpose:** Initializes HuggingFace models, manages CockroachDB connections, executes schema DDL, computes embeddings, and performs bulk data inserts.

- **Inputs:** DB credentials, Schema, Transformed Rows.

- **Outputs:** Number of inserted rows.

- **Key methods:** `get_model()`, `get_db_conn()`, `deploy_schema(schema)`, `embed_and_insert(...)`

### h. QueryOptimizer

- **File:** `core/optimizer.py`

- **Purpose:** Synthesizes test queries to check database performance and automatically adds indexes.

- **Inputs:** Database schema, Data profile.

- **Outputs:** Logs optimization actions; no direct return value.

- **Key methods:** `synthesize_and_tune(schema, profile)`

### i. IntelligentChatAgent

- **File:** `core/chat_agent.py`

- **Purpose:** Serves as a RAG interface. Classifies user intent, generates SQL queries, performs vector distance searches, and formats answers.

- **Inputs:** User query, Database URL, conversation history.

- **Outputs:** JSON response containing natural language answers and optional graph URLs.

- **Key methods:** `handle_agent_chat(...)`, `_clafy_intent(...)`, `_format_results_with_llm(...)`

## 5. External Services

- **Amazon S3:** Used for triggering the pipeline (via Event Notifications), downloading raw data, storing agent state (`agent_memory_state.txt`), storing EDA plots (`eda-output/`), and uploading serialized models.

- **CockroachDB Cloud:** Provisions serverless clusters via the `ccloud` CLI inside the AWS Lambda environment. Connects using `psycopg2`.

- **Groq API:** Primary LLM provider via the `openai` Python client. Used for fast schema generation, query synthesis, chat completions, and python code generation.

- **OpenRouter API:** Used specifically for the `nvidia/llama-3.1-nemotron-ultra-253b-v1:free` model in the `AutoMLAgent` reasoning loop to review ML code.

## 6. Database Architecture

The database implementation relies entirely on CockroachDB (PostgreSQL wire-compatible).

- **Schema:** The `SchemaArchitect` is strictly prompted to create ONE denormalized flat table containing all columns from the dataset.

- **Vectors:** An `embedding VECTOR(768)` column is appended to the table.

- **Indexes:** A `CREATE VECTOR INDEX` statement is executed alongside the table creation. Additional standard B-Tree indexes are created dynamically by the `QueryOptimizer` if EXPLAIN plans indicate full table scans.

- **Constraints:** Primary keys are generated synthetically or mapped from columns. `ON CONFLICT DO NOTHING` is used during inserts. Case-sensitivity is strictly enforced by double-quoting all identifiers (`"ColumnName"`).

## 7. LLM Integration

The repository uses the standard `openai` python package configured with custom `base_url`s:

- **Groq (`llama-3.3-70b-versatile` / `groq/compound`):** Used universally across `SchemaArchitect`, `QueryOptimizer`, `AutoMLAgent` (for writing code), and `IntelligentChatAgent`.

- **DeepSeek (`deepseek-chat`):** Referenced in `IntelligentChatAgent` for formatting results.

- **OpenRouter (`nvidia/llama-3.1-nemotron-ultra...`):** Used strictly as a "Reasoner" in `AutoMLAgent` to critique executed machine learning scripts and suggest improvements based on stack traces and metrics.

## 8. Vector Search

- **Embedding Generation:** Uses `sentence-transformers` with the `all-mpnet-base-v2` model. The model calculates embeddings locally within the AWS Lambda.

- **Format:** Row data is concatenated into a single string (e.g., `Col1: Val1 | Col2: Val2`) before being encoded.

- **Retrieval:** The `IntelligentChatAgent` embeds the user's natural language query and executes a SQL query using the `<->` KNN distance operator (e.g., `ORDER BY embedding <-> '[...]' LIMIT 5`) to retrieve semantically relevant context for RAG.

## 9. Query Optimization

The `QueryOptimizer` module autonomously manages database performance:

1. Synthesizes 3 realistic analytical queries using an LLM.

2. Runs `EXPLAIN {query}` on the database.

3. Parses the EXPLAIN output looking for the string `"full table scan"`.

4. If found, sends the EXPLAIN plan back to the LLM to generate `CREATE INDEX` statements.

5. Executes the returned indexing DDL directly on the database.

## 10. Limitations / Important Implementation Notes

- **Lambda Resource Constraints:** The pipeline bundles PyTorch, HuggingFace models, pandas, and scikit-learn. AWS Lambda requires high memory (e.g., 4096MB+) and a maximum timeout (15 minutes) to prevent Out-Of-Memory (OOM) or timeout crashes during the encoding phase.

- **Local vs Cloud Execution:** The `ccloud` cluster provisioning will only execute if the `TEST_DB_URL` environment variable is not present.

- **Data Inserts:** `VectorEngine.embed_and_insert` constructs `INSERT INTO` queries using the raw CSV headers. The implementation currently caps embedding processing to the first 500 rows (`transformed_rows[:500]`) to maintain speed limits for RAG demonstrations.

- **Dynamic Code Execution:** The `AutoMLAgent` uses the python `exec()` function to dynamically execute LLM-generated code. This relies on the isolated execution environment of AWS Lambda for security.
