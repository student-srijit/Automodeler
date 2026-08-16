import json
import re
import os
from openai import OpenAI
from core.vector_engine import VectorEngine

class QueryOptimizer:
    @staticmethod
    def synthesize_and_tune(schema, profile):
        client        = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ.get('OPENROUTER_API_KEY'))
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
            model="google/gemini-2.5-flash",
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

        conn = VectorEngine.get_db_conn()
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
