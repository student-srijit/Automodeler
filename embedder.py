import os
import psycopg2
import uuid
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

def ingest_data():
    print("Loading 768-dimension embedding model...")
    # all-mpnet-base-v2 naturally outputs a 768-dimensional vector
    model = SentenceTransformer('all-mpnet-base-v2') 
    
    # Sample data mimicking your CSV rows
    sample_data = [
        {"user_name": "ABC", "score": 98.5, "event_count": 42},
        {"user_name": "XYZ", "score": 95.0, "event_count": 38},
        {"user_name": "Rimo", "score": 88.2, "event_count": 15}
    ]
    
    db_url = os.getenv("DATABASE_URL")
    
    try:
        print("Connecting to CockroachDB...")
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        
        with conn.cursor() as cur:
            for row in sample_data:
                # Create a rich text representation for the neural network to analyze
                text_to_embed = f"User {row['user_name']} has a score of {row['score']} with {row['event_count']} events."
                
                print(f"Generating vector for: {row['user_name']}")
                # Generate the embedding and convert to a standard Python list
                vector = model.encode(text_to_embed).tolist()
                
                # Insert the operational data AND the vector into the same row
                insert_query = """
                    INSERT INTO sample_table (id, user_name, score, event_count, created_at, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                cur.execute(insert_query, (
                    str(uuid.uuid4()), 
                    row['user_name'], 
                    row['score'], 
                    row['event_count'], 
                    datetime.now(), 
                    str(vector) # CockroachDB expects the vector as a string representation of the array
                ))
                
        conn.close()
        print("\nSUCCESS: Operational data and deep learning vectors successfully embedded and stored!")
        
    except Exception as e:
        print(f"\nIngestion failed: {e}")

if __name__ == "__main__":
    ingest_data()
