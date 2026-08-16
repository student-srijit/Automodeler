# AutoModeler Serverless Setup Guide

This guide walks you through deploying the Dual-Mode Agent architecture to AWS Lambda and fetching the necessary API keys.

---

## 🔑 1. Environment Variables & Keys

To run the agent (either locally or on AWS), you will need these keys in your environment or `.env` file:

### CockroachDB Cloud (`CCLOUD_API_KEY`)
Used by the DevOps agent to autonomously spin up new clusters.
1. Log into your [CockroachDB Cloud Console](https://cockroachlabs.cloud/).
2. On the left sidebar, click **Access Management** -> **Service Accounts**.
3. Create a Service Account (e.g., `agent-admin`) and assign it the **Cluster Admin** or **Developer** role.
4. Click **Generate API Key**. Copy this immediately!

### Groq AI (`GROQ_API_KEY`)
Used by the Chat agent for fast Llama 3.3 inference.
1. Go to [GroqCloud](https://console.groq.com/keys).
2. Create an account and click **Create API Key**.

### Local Database Override (`TEST_DB_URL`) *(Optional)*
Used ONLY if you are testing locally and want to bypass the `ccloud` cluster creation.
1. In CockroachDB Cloud, click **Clusters** -> **Create Cluster** (Serverless).
2. Once created, click **Connect**.
3. Copy the `postgresql://...` connection string.
4. **Important**: Go to the **Networking** tab for that cluster and add `0.0.0.0/0` (or your IP) so your local machine can connect!

### AWS Keys (`AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY`)
Used by your local testing scripts (`local_server.py`) to access S3. *Not required when deployed to Lambda.*
1. In the AWS Console, search for **IAM**.
2. Go to **Users**, select your user (e.g., `automodeler-admin`).
3. Click the **Security credentials** tab.
4. Scroll to **Access keys** and click **Create access key**.

---

## 🚀 2. AWS Lambda Deployment

### Step A: Push the Code to AWS ECR
Your AWS Lambda function runs inside a custom Docker container.
1. Open your terminal and navigate to your project: `cd /Users/kausheyaroy/Desktop/MyProjects/Automodeler`
2. Go to **ECR** in the AWS Console and click your `automodeler-lambda` repository.
3. Click the **View push commands** button.
4. **Command 1:** Run the login command.
5. **Command 2 (Build):** *DO NOT use the default AWS command.* Instead, run:
   `docker build --provenance=false -t automodeler-lambda .`
6. **Command 3:** Run the tag command.
7. **Command 4:** Run the push command.

### Step B: Tell Lambda to Use the New Code
1. Go to your Lambda function in the AWS Console.
2. Click the **Image** tab.
3. Click **Deploy new image**.
4. Click **Browse images**, select your repository, and select the newly uploaded image (look at the timestamp).
5. Click **Save**.

---

## ⚙️ 3. AWS Lambda Configuration

### Memory & Timeout (Crucial for AI)
1. Go to the **Configuration** tab -> **General configuration** -> **Edit**.
2. Set Memory to **3008 MB**.
3. Set Timeout to **15 min 0 sec**.

### Environment Variables
1. Go to the **Configuration** tab -> **Environment variables** -> **Edit**.
2. Add your keys: `GROQ_API_KEY`, `CCLOUD_API_KEY`, etc.

### Permissions
1. Go to the **Configuration** tab -> **Permissions**.
2. Click the execution role link (opens IAM).
3. Click **Add permissions** -> **Attach policies**.
4. Add **AmazonS3FullAccess** (so the agent can read CSVs and write its memory state).

### The S3 Trigger
1. At the top of your Lambda page, click **+ Add trigger**.
2. Select **S3** and choose your bucket (`automodeler-uploads...`).
3. Set Suffix to `.csv`.
4. Click **Add**.

---

## 💻 4. Using the Agent
1. Open `index.html` in your browser.
2. Paste your **API Gateway URL** and **S3 Bucket Name**.
3. Upload a CSV file!
4. Behind the scenes, S3 triggers Lambda, Lambda provisions CockroachDB, embeds the data, and saves the connection string back to S3.
5. Ask a question in the chat!