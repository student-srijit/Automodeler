import json
import subprocess
import os

class ClusterProvisioner:
    @staticmethod
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

    @staticmethod
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
