import os
import psycopg2

_model = None

class VectorEngine:
    @staticmethod
    def get_model():
        global _model
        if _model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model (first cold start)...")
            model_path = os.environ.get('MODEL_PATH', 'all-mpnet-base-v2')
            _model = SentenceTransformer(model_path)
        return _model

    @staticmethod
    def get_db_conn():
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        conn.autocommit = True
        return conn

    @staticmethod
    def deploy_schema(schema):
        conn = VectorEngine.get_db_conn()
        with conn.cursor() as cur:
            for tbl_def in schema["tables"]:
                tbl_name = tbl_def['table_name']
                print(f"Ensuring table exists: {tbl_name}")
                # Drop the specific table being replaced to ensure a clean slate for this upload
                cur.execute(f'DROP TABLE IF EXISTS "{tbl_name}" CASCADE;')
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

    @staticmethod
    def embed_and_insert(schema, headers, transformed_rows, batch_size=500):
        model         = VectorEngine.get_model()
        conn          = VectorEngine.get_db_conn()
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
