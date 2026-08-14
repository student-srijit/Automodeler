# AutoModeler: Serverless Dual-Mode RAG Agent

AutoModeler is an autonomous, event-driven AI agent that takes a raw CSV file, dynamically provisions a CockroachDB Serverless database in the cloud, generates a normalized schema, computes vector embeddings, and instantly provides a conversational interface to query the data—all without human intervention.

## Version 2.0

We completely transformed a basic script into a **production-ready Serverless Architecture**:

1. **Dual-Mode Event-Driven Architecture:** Converted the script into an AWS Lambda function. It now listens for S3 `ObjectCreated` events to trigger the background ETL, and API Gateway events to trigger real-time chat.
2. **Decentralized State Memory:** The DevOps agent dynamically generates a new CockroachDB cluster and securely saves the new `DATABASE_URL` into an S3 file (`agent_memory_state.txt`). The Chat agent dynamically fetches this state, making the Lambda functions entirely stateless!
3. **Dynamic EXPLAIN Plan Parsing:** Instead of statically listing table indexes, the AI agent literally queries the CockroachDB query optimizer (`EXPLAIN SELECT...`) to show you the *exact execution plan* (e.g., `Lookup Join -> Vector KNN Search`) used for every individual question.
4. **Local Testing Simulation Suite:** Created `local_server.py` and `trigger_etl.py` to perfectly simulate AWS S3 and API Gateway locally on your Mac, so you can test the cloud architecture without actually deploying it.
5. **Modern Glassmorphism UI:** Built a highly polished, responsive frontend (`index.html`) with beautiful typography, Markdown rendering (`marked.js`), and LocalStorage persistence.

## Architecture Pipeline

```text
[User / Browser] -- (CSV Upload) --> [AWS S3 Bucket]
                                        | (Triggers Event)
                                        v
                            [AWS Lambda: DevOps Agent]
                                /                  \
                      (ccloud CLI)               (ETL Pipeline)
                      /                              \
        [CockroachDB Serverless] <--- (Schema, Data, Vectors)
                  |
        (Saves DB URL State)
                  |
                  v
[AWS S3 Bucket: agent_memory_state.txt]

[User / Browser] -- (Chat Query) --> [API Gateway] -> [AWS Lambda: Chat Agent]
                                                           |
                                                (Fetches State from S3)
                                                           |
                                               (Queries CockroachDB KNN)
                                                           |
                                               (Groq Llama-3.3 LLM RAG)
                                                           |
[User / Browser] <-----------------------------------------+ (Formatted Markdown Response)
```

---

## 1. How to Run & Test Locally

You can run the entire serverless architecture directly on your laptop using the provided testing scripts.

### Prerequisites
1. Ensure your `.env` file contains:
   - `GROQ_API_KEY`
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `TEST_DB_URL` (Optional: If you want to bypass the `ccloud` creation and use an existing CockroachDB cluster. *Ensure your IP is allowlisted in the DB network settings!*)

### Step 1: Start the Local API Gateway Simulator
In your first terminal, run:
```sh
python3 local_server.py
```
This runs a Flask server on `http://127.0.0.1:8000` that acts just like AWS API Gateway.

### Step 2: Trigger the S3 Upload Event

In a second terminal, run:
```sh
python3 trigger_etl.py
```
This script acts exactly like an AWS S3 trigger. It reads `large_sample.csv`, runs the heavy AI ETL pipeline, uploads the data to CockroachDB, and saves the agent's memory state back to S3. **Wait for it to print `=== PIPELINE FINISHED SUCCESSFULLY ===`.**

### Step 3: Test the UI
1. Open `index.html` in your web browser.
2. **Lambda API Gateway URL:** `http://127.0.0.1:8000/api`
3. **AWS S3 Bucket Name:** `automodeler-uploads-rimo-1234` (or your actual bucket name)
4. *Skip the CSV upload section (since you just manually ran the trigger script).*
5. Start asking questions in the chat!

---

##  2. How to Deploy to AWS

*(For a detailed walkthrough of getting API keys, see `setup.md`)*

1. **Push the Docker Container:** Use the AWS ECR push commands to upload the code to your `automodeler-lambda` repository.
   - *Important:* Use `docker build --provenance=false -t automodeler-lambda .` to avoid image index errors.
2. **Update Lambda:** Go to your Lambda function -> **Image** tab -> **Deploy new image**.
3. **Configure Lambda:**
   - Memory: `3008 MB`
   - Timeout: `15 minutes`
   - Add your Environment Variables (`GROQ_API_KEY`, `CCLOUD_API_KEY`).
   - Give the Execution Role `AmazonS3FullAccess`.
4. **Add the S3 Trigger:** Add a trigger for your S3 bucket with the `.csv` suffix.

---

## 3. How to Test the AWS Cloud Deployment

Once deployed to AWS, the system is 100% autonomous. You do not need to use the terminal.

1. Open `index.html`.
2. Enter your real **AWS API Gateway URL** and your **S3 Bucket Name**.
3. **Upload the CSV:** Select `large_sample.csv` and click "Upload & Analyze".
4. **Wait:** The browser will upload it to S3, which silently wakes up Lambda in the background. Wait a few minutes for the cluster to provision and the embeddings to generate.
5. **Ask Questions!**

### Sample Questions to Ask:
- *"What is the absolute most expensive electronic item you currently have in stock?"*
- *"I'm looking for a highly durable accessory for sports. Do you have anything like that, and how much does it cost?"* (Tests Semantic Vector Search!)
- *"Can you recommend a highly-rated book that costs less than $100?"*
- *"Can you list any premium widgets that are currently out of stock? I want to know their prices."*

Don't forget to look at the green **Metrics Badge** under the bot's response to see the exact CockroachDB Query Execution Plan!