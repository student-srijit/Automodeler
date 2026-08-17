import json
import re
import os
from openai import OpenAI

class SchemaArchitect:
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

    @staticmethod
    def generate_schema(profile):
        client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ.get('OPENROUTER_API_KEY'))
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SchemaArchitect.SCHEMA_PROMPT},
                {"role": "user",   "content": f"Generate normalized schema:\n{json.dumps(profile)}"}
            ],
            model="google/gemini-2.5-flash",
            temperature=0.1,
            max_tokens=4096
        )
        content = response.choices[0].message.content.strip()
        content = re.sub(r'^```[a-z]*\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'```\s*$', '', content, flags=re.MULTILINE)
        schema  = json.loads(content.strip())
        print("STAGE 2 SCHEMA:", json.dumps(schema, indent=2))
        return schema
