import json
import urllib.parse
import os
import traceback

from core.profiler import DataProfiler
from core.provisioner import ClusterProvisioner
from core.architect import SchemaArchitect
from core.transformer import DataTransformer
from core.vector_engine import VectorEngine
from core.optimizer import QueryOptimizer
from core.utils import s3

class AutoModelerPipeline:
    @staticmethod
    def run_s3_etl(event):
        bucket = event['Records'][0]['s3']['bucket']['name']
        key    = urllib.parse.unquote_plus(
            event['Records'][0]['s3']['object']['key'], encoding='utf-8'
        )
        print(f"\n{'='*60}")
        print(f"AutoModeler Agent ETL — s3://{bucket}/{key}")
        print(f"{'='*60}\n")

        try:
            if os.environ.get('TEST_DB_URL'):
                print(">>> Using TEST_DB_URL from environment for local testing...")
                new_db_url = os.environ.get('TEST_DB_URL').replace("sslmode=verify-full", "sslmode=require")
                os.environ['DATABASE_URL'] = new_db_url
            else:
                new_db_url = ClusterProvisioner.provision_agent_cluster()
            
            print(f">>> Saving agent memory state to s3://{bucket}/agent_memory_state.txt")
            s3.put_object(Bucket=bucket, Key="agent_memory_state.txt", Body=new_db_url.encode('utf-8'))
            s3.put_object(Bucket=bucket, Key="agent_filename.txt", Body=key.encode('utf-8'))
            
            print(">>> STAGE 1: Advanced Data Profiling...")
            profile, headers, rows, dupe_idx = DataProfiler.profile_csv(bucket, key, s3)

            print(">>> STAGE 3: Cluster Sizing...")
            cluster_plan = ClusterProvisioner.size_cluster(profile)

            print(">>> STAGE 2: AI Schema Generation (multi-table normalized)...")
            schema = SchemaArchitect.generate_schema(profile)

            print(">>> STAGE 4: CockroachDB Provisioning (multi-table DDL)...")
            VectorEngine.deploy_schema(schema)

            print(">>> STAGE 5: Data Transformation (clean + impute + deduplicate)...")
            transformed = DataTransformer.transform_rows(headers, rows, profile["columns"], dupe_idx)

            print(">>> STAGE 6: Batch Embedding & Load...")
            rows_inserted = VectorEngine.embed_and_insert(schema, headers, transformed)

            print(">>> STAGE 7: Autonomous Query Testing & Index Optimization...")
            QueryOptimizer.synthesize_and_tune(schema, profile)

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
