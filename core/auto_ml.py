import os
import json
import time
import re
import boto3
from openai import OpenAI
from core.utils import stream_status, detect_modality, s3, log_experiment_to_cockroach, publish_model_to_s3

class AutoMLAgent:
    @staticmethod
    def call_nemotron_reasoner(code, execution_logs, metric, target_column, round_num, approach_history):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get('OPENROUTER_API_KEY', ''),
            default_headers={"HTTP-Referer": "https://automodeler.ai", "X-Title": "AutoModeler"}
        )
        prompt = f"""You are an elite ML Systems Reasoning Expert conducting a code review.
PIPELINE ROUND: {round_num}/3
TARGET COLUMN: {target_column}
APPROACHES TRIED SO FAR:
{chr(10).join(approach_history) if approach_history else 'None yet.'}

=== GENERATED ML CODE ===
{code[:6000]}

=== EXECUTION OUTPUT ===
{execution_logs[:3000]}

=== METRIC ACHIEVED ===
{metric}

Perform deep reasoning on this pipeline:
1. Are the hyperparameters optimal? (n_neighbors, C, n_estimators, etc.)
2. Is the feature engineering appropriate for the data type?
3. Is the algorithm the best choice, or would a different one score higher?
4. Are there bugs, data leakage, or implementation inefficiencies?
5. If metric is N/A or execution crashed, what specifically caused it?

Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):
{{
  "verdict": "accept" | "improve" | "new_approach",
  "reasoning": "your concise but thorough analysis here",
  "suggestions": "specific, concrete code-level improvements to apply",
  "new_algorithm": "if verdict is new_approach: describe the completely different algorithm and strategy to try instead"
}}

Verdict rules:
- "accept": metric is strong and code quality is good
- "improve": same core approach, but fix hyperparameters / feature engineering / bugs
- "new_approach": fundamentally different algorithm needed (use only in round 1 or 2)
"""
        print(f">>> [Nemotron Reasoner] Calling nvidia/llama-3.1-nemotron-ultra-253b-v1:free ...")
        resp = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.2
        )
        raw = resp.choices[0].message.content.strip()
        print(f">>> [Nemotron] Raw response: {raw[:500]}")
        json_match = re.search(r'\{[\s\S]*?\}(?=\s*$|\s*\n)', raw)
        if not json_match:
            json_match = re.search(r'\{[\s\S]+\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except Exception:
                pass
        verdict = "improve"
        if "accept" in raw.lower()[:200]:
            verdict = "accept"
        elif "new_approach" in raw.lower()[:200] or "new approach" in raw.lower()[:200]:
            verdict = "new_approach"
        return {"verdict": verdict, "reasoning": raw[:1000], "suggestions": raw[500:1500], "new_algorithm": ""}

    @staticmethod
    def _parse_metric_value(metric_str):
        if not metric_str or metric_str in ("N/A", "Metric output not found.", "Ready"):
            return -1.0
        try:
            nums = re.findall(r'[\d]*\.?[\d]+', str(metric_str))
            if nums:
                val = float(nums[0])
                return val / 100.0 if val > 1.5 else val
        except Exception:
            pass
        return -1.0

    @staticmethod
    def run_reasoning_loop(analyst_prompt, coder_extra_context, bucket, start_time, target_column="target", max_rounds=3, job_id=None):
        import io, sys
        groq_client = OpenAI(base_url='https://api.groq.com/openai/v1', api_key=os.environ.get('GROQ_API_KEY'))
        best_result = None
        best_metric_val = -1.0
        rounds_history = []
        approach_history = []
        current_analyst_prompt = analyst_prompt
        current_coder_extra = coder_extra_context
        
        for round_num in range(1, max_rounds + 1):
            print(f"\n{'='*60}")
            print(f">>> REASONING LOOP — ROUND {round_num}/{max_rounds}")
            print(f"{'='*60}")
            
            print(">>> [Agent 1 — Analyst] meta-llama/llama-3.3-70b-instruct (Groq)")
            stream_status(job_id, bucket, f"Agent 1 (Strategist): DeepSeek-R1 analyzing dataset for Round {round_num}...")
            try:
                analyst_response = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": current_analyst_prompt}],
                    model="groq/compound",
                    max_tokens=2048,
                    temperature=0.1
                )
                strategy_plan = analyst_response.choices[0].message.content.strip()
            except Exception as e:
                print(f">>> DeepSeek-R1 failed ({e}), falling back to LLaMA 3.3 70B")
                analyst_response = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": current_analyst_prompt}],
                    model="groq/compound",
                    max_tokens=2048,
                    temperature=0.1
                )
                strategy_plan = analyst_response.choices[0].message.content.strip()
            
            print(f">>> [Analyst] Strategy:\n{strategy_plan[:500]}...")
            approach_history.append(f"Round {round_num}: {strategy_plan[:150]}...")
            
            stream_status(job_id, bucket, f"Agent 2 (Coder): LLaMA-3.3 writing implementation for Round {round_num}...")
            coder_prompt = f"""You are an elite AI Coding Agent. Implement this ML strategy precisely.
Our Lead Data Scientist's strategy:
----------
{strategy_plan}
----------
{current_coder_extra}
CRITICAL REQUIREMENTS (follow in order):
1. Calculate a numeric performance metric and print "FINAL_METRIC: <numeric_value>" at the very end.
2. After training, serialize the FINAL trained model object to /tmp/model.pkl using joblib (preferred) or pickle.
   Example: import joblib; joblib.dump(model, '/tmp/model.pkl')
3. Output ONLY raw Python code. No markdown. No ``` blocks. Pure executable Python only.
"""
            print(">>> [Agent 2 — Coder] meta-llama/llama-3.3-70b-instruct (Groq)")
            coder_response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": coder_prompt}],
                model="groq/compound",
                max_tokens=4096,
                temperature=0.05
            )
            generated_code = coder_response.choices[0].message.content.strip()
            generated_code = re.sub(r'^```[a-z]*\n?', '', generated_code, flags=re.MULTILINE)
            generated_code = re.sub(r'```\s*$', '', generated_code, flags=re.MULTILINE)
            generated_code = generated_code.strip()
            print(f">>> [Coder] Code ({len(generated_code)} chars) generated.")
            
            print(">>> [Executor] Running generated code...")
            stream_status(job_id, bucket, f"Executing generated code for Round {round_num}...")
            old_stdout = sys.stdout
            redirected_output = sys.stdout = io.StringIO()
            exec_error = None
            try:
                exec(generated_code, {})
            except Exception as e:
                exec_error = str(e)
            finally:
                sys.stdout = old_stdout
            
            execution_output = redirected_output.getvalue()
            if exec_error:
                execution_output += f"\nEXECUTION_ERROR: {exec_error}"
                print(f">>> [Executor] Error: {exec_error}")
            
            metric_match = re.search(r'FINAL_METRIC:\s*(.+)', execution_output)
            final_metric = metric_match.group(1).strip() if metric_match else "N/A"
            metric_val = AutoMLAgent._parse_metric_value(final_metric)
            print(f">>> [Executor] FINAL_METRIC={final_metric} (numeric={metric_val:.4f})")
            
            if os.path.exists('/tmp/submission.csv'):
                try:
                    s3_client = boto3.client('s3')
                    s3_client.upload_file('/tmp/submission.csv', bucket, 'submission.csv')
                    execution_output += "\n[SYSTEM] submission.csv uploaded to S3!"
                    print(">>> [Executor] Uploaded submission.csv to S3")
                except Exception as e:
                    execution_output += f"\n[SYSTEM] S3 upload failed: {e}"
            
            round_data = {
                "round": round_num,
                "strategy": strategy_plan,
                "code": generated_code,
                "logs": execution_output,
                "metric": final_metric,
                "metric_val": metric_val,
                "nemotron_verdict": None,
                "nemotron_reasoning": None
            }
            rounds_history.append(round_data)
            
            if best_result is None or metric_val > best_metric_val:
                best_metric_val = metric_val
                best_result = round_data
                print(f">>> [Tracker] New best metric: {final_metric}")
            
            if round_num >= 2:
                prev_val = rounds_history[-2]["metric_val"]
                if prev_val > 0 and metric_val >= prev_val * 1.05:
                    print(f">>> [Loop] Early stop: metric improved {prev_val:.4f} → {metric_val:.4f} (>5%)")
                    round_data["nemotron_verdict"] = "early_stop"
                    break
            
            if round_num >= max_rounds:
                break
            
            print(f">>> [Agent 3 — Nemotron] Reviewing round {round_num} results...")
            stream_status(job_id, bucket, f"Agent 3 (Reviewer): Nemotron-253B evaluating Round {round_num} performance...")
            nemotron_review = {"verdict": "improve", "reasoning": "Reasoner unavailable.", "suggestions": "", "new_algorithm": ""}
            try:
                nemotron_review = AutoMLAgent.call_nemotron_reasoner(
                    generated_code, execution_output, final_metric,
                    target_column, round_num, approach_history
                )
            except Exception as e:
                print(f">>> [Nemotron] Call failed: {e}. Defaulting to 'improve'.")
            
            verdict = nemotron_review.get("verdict", "improve")
            stream_status(job_id, bucket, f"Agent 3 Verdict: {verdict.upper()}. Planning next step...")
            suggestions = nemotron_review.get("suggestions", "")
            new_algorithm = nemotron_review.get("new_algorithm", "")
            reasoning = nemotron_review.get("reasoning", "")
            
            round_data["nemotron_verdict"] = verdict
            round_data["nemotron_reasoning"] = reasoning
            print(f">>> [Nemotron] Verdict: {verdict.upper()}")
            
            if verdict == "accept":
                print(">>> [Loop] Nemotron accepted result. Stopping early.")
                break
            elif verdict == "improve":
                current_coder_extra = f"""{coder_extra_context}

REVISION INSTRUCTIONS (from Reasoning Agent — Round {round_num} review):
Previous metric: {final_metric}
Apply these specific improvements to the code:
{suggestions}

The previous code had these issues:
{reasoning[:800]}
"""
            elif verdict == "new_approach":
                current_analyst_prompt = f"""{analyst_prompt}

IMPORTANT CONTEXT — ROUND {round_num} FAILED (metric: {final_metric}):
A reasoning model (Nvidia Nemotron 253B) reviewed the previous approach and recommends
switching to a COMPLETELY DIFFERENT algorithm/strategy:
{new_algorithm}

Previous reasoning:
{reasoning[:800]}

Design a fresh strategy based on the recommended approach above.
"""
                current_coder_extra = coder_extra_context
                print(f">>> [Loop] Switching to new approach for Round {round_num + 1}.")
        
        latency_ms = int((time.time() - start_time) * 1000)
        best = best_result or rounds_history[0]
        
        improvement_history = [
            {
                "round": r["round"],
                "metric": r["metric"],
                "verdict": r.get("nemotron_verdict") or "final",
                "reasoning_summary": (r.get("nemotron_reasoning") or "")[:200]
            }
            for r in rounds_history
        ]
        
        print(f"\n>>> REASONING LOOP COMPLETE: {len(rounds_history)} rounds, best metric={best['metric']}")
        
        print(">>> [Integration] Publishing model to S3 Registry...")
        s3_model_url = publish_model_to_s3(bucket, target_column)
        
        print(">>> [Integration] Logging experiment to CockroachDB Tracker...")
        log_experiment_to_cockroach(
            target_column=target_column,
            final_metric=best["metric"],
            metric_value=best.get("metric_val", 0),
            rounds_taken=len(rounds_history),
            s3_model_url=s3_model_url,
            reasoning_summary=best.get("nemotron_reasoning", "Accepted without review.")
        )
        
        return {
            "status": "success",
            "final_metric": best["metric"],
            "generated_code": best["code"],
            "execution_logs": best["logs"],
            "metrics": {"latency_ms": latency_ms},
            "rounds_taken": len(rounds_history),
            "improvement_history": improvement_history,
            "nemotron_reasoning": best.get("nemotron_reasoning") or ""
        }

    @staticmethod
    def handle_image_task(bucket, filename, target_column, job_id=None):
        print(f">>> [IMAGE MODALITY] file={filename} target={target_column}")
        start_time = time.time()
        analyst_prompt = f"""You are an elite AI Lead Data Scientist specializing in Computer Vision.
The user has uploaded an IMAGE dataset named '{filename}' to an S3 bucket '{bucket}'.
The target task is: '{target_column}'.

Typical structure: a .zip file containing sub-folders per class (e.g. cats/, dogs/), or a CSV manifest with columns [filename, label].

Develop an optimal Computer Vision ML strategy:
1. How to load the images from the zip (use zipfile + Pillow).
2. Feature extraction: use a pre-trained ResNet-50 (torchvision, remove final FC layer) as a feature extractor — no GPU needed.
3. Normalize features. For classification use SVM or LogisticRegression. For similarity/retrieval use NearestNeighbors.
4. If test images exist (a test.zip or test folder), generate predictions and save to submission.csv with columns [filename, {target_column}] for classification, or [filename, Index_list] for retrieval.
5. Print FINAL_METRIC: <accuracy or distance>.

Write a clear step-by-step algorithmic blueprint. Do NOT write Python code yet.
"""
        coder_context = f"""
Assume `{filename}` is available locally (already downloaded from S3).
Use only: zipfile, os, Pillow (PIL), numpy, sklearn, and optionally torch+torchvision (CPU only).
Save output to `submission.csv`.
"""
        result = AutoMLAgent.run_reasoning_loop(analyst_prompt, coder_context, bucket, start_time, target_column=target_column, job_id=job_id)
        result['modality'] = 'image'
        return result

    @staticmethod
    def handle_audio_task(bucket, filename, target_column, job_id=None):
        print(f">>> [AUDIO MODALITY] file={filename} target={target_column}")
        start_time = time.time()
        analyst_prompt = f"""You are an elite AI Lead Data Scientist specializing in Audio ML.
The user has uploaded an AUDIO dataset named '{filename}' to an S3 bucket '{bucket}'.
The target task is: '{target_column}'.

Typical structure: a .zip file containing sub-folders per class (e.g. happy/, sad/), each with .wav or .mp3 files.

Develop an optimal Audio ML strategy:
1. How to unzip and load audio files using librosa.
2. Feature extraction: extract MFCC features (n_mfcc=40) + chroma + spectral contrast from each file. Average across time axis.
3. Normalize features with StandardScaler.
4. For classification: SVM or RandomForest. For retrieval: NearestNeighbors (cosine).
5. If test audio exists (test.zip or test folder), generate predictions and save to submission.csv with columns [filename, {target_column}].
6. Print FINAL_METRIC: <accuracy or distance>.

Write a clear step-by-step algorithmic blueprint. Do NOT write Python code yet.
"""
        coder_context = f"""
Assume `{filename}` is available locally.
Use only: zipfile, os, numpy, librosa, soundfile, sklearn.
Save output to `submission.csv`.
"""
        result = AutoMLAgent.run_reasoning_loop(analyst_prompt, coder_context, bucket, start_time, target_column=target_column, job_id=job_id)
        result['modality'] = 'audio'
        return result

    @staticmethod
    def handle_automl_train(target_column, db_url, bucket, filename, job_id=None):
        print(f">>> Handling AutoML training for target: {target_column}")
        start_time = time.time()

        if db_url:
            os.environ["DATABASE_URL"] = db_url

        modality = detect_modality(filename)
        print(f">>> Detected modality: {modality.upper()} for file: {filename}")

        if modality == 'image':
            result = AutoMLAgent.handle_image_task(bucket, filename, target_column, job_id)
            result['target_column'] = target_column
            return result

        if modality == 'audio':
            result = AutoMLAgent.handle_audio_task(bucket, filename, target_column, job_id)
            result['target_column'] = target_column
            return result

        import pandas as pd
        sample_data = []
        columns = []
        try:
            try:
                obj = s3.get_object(Bucket=bucket, Key=filename)
                df_sample = pd.read_csv(obj['Body'], nrows=10)
            except:
                df_sample = pd.read_csv(filename, nrows=10)
            columns = df_sample.columns.tolist()
            for _, row in df_sample.iterrows():
                row_dict = row.to_dict()
                for k, v in row_dict.items():
                    if pd.isna(v):
                        row_dict[k] = None
                    elif type(v).__name__ in ['datetime', 'date', 'Timestamp']:
                        row_dict[k] = str(v)
                sample_data.append(row_dict)
        except Exception as e:
            return {"error": f"Failed to load sample data from {filename}: {e}"}

        if target_column not in columns:
            return {"error": f"Target column '{target_column}' not found in dataset. Available columns: {columns}"}

        analyst_prompt = f"""You are an elite AI Lead Data Scientist.
Your task is to define the optimal ML mathematical strategy for the target column '{target_column}'.

Here is a sample of the data:
{json.dumps(sample_data, indent=2)}

REQUIREMENTS FOR YOUR STRATEGY:
1. The dataset will be `/tmp/train.csv` and `/tmp/test.csv`.
2. We need to find the top 10 most similar rows in the training dataset for each row in the test dataset.
3. Suggest the optimal feature extraction (e.g., TfidfVectorizer for text, StandardScaler for numeric).
4. Suggest the optimal model (sklearn.neighbors.NearestNeighbors) to find the 10 nearest neighbors.
5. The training set has an `Index` column. The output should map to training `Index` values, not array indices.
6. Save to `/tmp/submission.csv` with columns `Index` (from test) and `Index_list` (python list of 10 `Index` ints from train).

Write a clear, detailed algorithmic blueprint. Do NOT write Python code yet.
"""
        coder_context = f"""
Write a standalone Python script using scikit-learn that implements this strategy.
Assume `train.csv` and `test.csv` are located at `/tmp/train.csv` and `/tmp/test.csv`.
Save the output to `/tmp/submission.csv`.
CRITICAL RULES:
1. Do NOT drop the '{target_column}' column. You likely need it to generate features.
2. If you must drop columns, always use `errors='ignore'` (e.g. `df.drop(columns=['...'], errors='ignore')`).
3. Only use feature columns that exist in BOTH `train.csv` and `test.csv`.
4. IMPORTANT FOR SPEED: You MUST sample the training data using `train_df = train_df.head(2000)` immediately after reading it to prevent execution timeouts!
Column reference (first 10 rows sample):
{json.dumps(sample_data, indent=2)}
"""
        try:
            s3.download_file(bucket, "train.csv", "/tmp/train.csv")
            s3.download_file(bucket, "test.csv", "/tmp/test.csv")
        except Exception as e:
            print(f"Warning: Failed to download train/test CSVs to /tmp: {e}")

        result = AutoMLAgent.run_reasoning_loop(analyst_prompt, coder_context, bucket, start_time, target_column=target_column, job_id=job_id)
        result['target_column'] = target_column
        result['modality'] = 'tabular'
        return result
