import os
import json
import psycopg2
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()
client = Groq()

def load_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def deploy_to_cockroach(schema_json):
    print("\nConnecting to CockroachDB Cloud...")
    db_url = os.getenv("DATABASE_URL")
    
    try:
        # Connect using psycopg2
        conn = psycopg2.connect(db_url)
        # Autocommit is highly recommended for CockroachDB schema changes
        conn.autocommit = True  
        
        with conn.cursor() as cur:
            table_name = schema_json['table_name']
            
            # Add this line to clear out the old table from previous runs
            print(f"Cleaning up old schema (DROP TABLE IF EXISTS {table_name})...")
            cur.execute(f"DROP TABLE IF EXISTS {table_name};")
            
            print("Executing CREATE TABLE...")
            cur.execute(schema_json['create_table_sql'])
            
            print("Executing CREATE VECTOR INDEX...")
            cur.execute(schema_json['create_vector_index_sql'])
            
        conn.close()
        print("\nSUCCESS: Database schema and vector index deployed!")
        
    except Exception as e:
        print(f"\nDeployment failed: {e}")

def generate_schema():
    print("Loading data profile and system prompt...")
    data_profile = load_file("data/sample_profile.json")
    system_prompt = load_file("prompts/system_prompt.txt")

    print("Sending payload to Groq Swarm Orchestrator...")
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate the CockroachDB schema for the following data profile:\n{data_profile}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
        response_content = chat_completion.choices[0].message.content
        print("\n--- GROQ RAW OUTPUT ---")
        print(response_content)
        
        # Strip markdown code fences if the LLM wraps the JSON in ```json ... ```
        cleaned = response_content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]  # drop the opening ```json line
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]  # drop the closing ```
        cleaned = cleaned.strip()
        
        # Verify it parses as JSON
        schema_json = json.loads(cleaned)
        
        # Trigger Phase 3: The automated deployment
        deploy_to_cockroach(schema_json)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_schema()
