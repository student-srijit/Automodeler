import json
import re
import os
import time
from openai import OpenAI
from core.vector_engine import VectorEngine

class IntelligentChatAgent:
    @staticmethod
    def _get_table_schema(conn, table_name):
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            return {row[0]: row[1] for row in cur.fetchall() if row[0] != 'embedding'}

    @staticmethod
    def _safe_sql_execute(conn, sql, params=None):
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                conn.commit()
                if cur.description:
                    cols = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    if 'embedding' in cols:
                        emb_idx = cols.index('embedding')
                        cols.pop(emb_idx)
                        rows = [row[:emb_idx] + row[emb_idx+1:] for row in rows]
                    serialized_rows = []
                    for row in rows:
                        serialized_rows.append([
                            str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                            for v in row
                        ])
                    return cols, serialized_rows, None
                else:
                    return [], [], None
        except Exception as e:
            return None, None, str(e)

    @staticmethod
    def _build_messages(system_prompt, history, user_query):
        msgs = [{"role": "system", "content": system_prompt}]
        if history:
            for h in history:
                role = "user" if h.get("role") == "user" else "assistant"
                content = h.get("content", "")
                if content:
                    msgs.append({"role": role, "content": content})
        msgs.append({"role": "user", "content": user_query})
        return msgs

    @staticmethod
    def _api_call_with_retry(client, messages, model, temperature, max_tokens, response_format=None):
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "messages": messages,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                if response_format:
                    kwargs["response_format"] = response_format
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                if "429" in str(e):
                    print(f"Rate limited. Retrying attempt {attempt+1}/{max_retries}...")
                    time.sleep(2)
                    if attempt == max_retries - 1:
                        raise e
                else:
                    raise e

    @staticmethod
    def _classify_intent(groq_client, user_query, history):
        resp = IntelligentChatAgent._api_call_with_retry(
            client=groq_client,
            messages=IntelligentChatAgent._build_messages("""You are an intent classifier for an AI data platform. Classify the user message into exactly one of:
- "train": user wants to predict, train, or build a model for a specific column
- "clean_action": user is responding to a data cleaning suggestion (approving, rejecting, or specifying a method)
- "eda": user wants to explore data, ask statistics, find missing values, get distributions, etc.

Respond ONLY with a valid JSON object and nothing else. Examples:
{"intent": "train", "target": "Price"}
{"intent": "eda"}
{"intent": "clean_action", "confirm": true, "method": "mean"}
{"intent": "clean_action", "confirm": false}""", history, user_query),
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=100
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
        try:
            return json.loads(raw.strip())
        except Exception:
            if any(w in user_query.lower() for w in ['predict', 'train', 'model', 'classify', 'forecast']):
                words = user_query.split()
                for i, w in enumerate(words):
                    if w.lower() in ('predict', 'for', 'target', 'classify'):
                        if i + 1 < len(words):
                            return {"intent": "train", "target": words[i+1].strip("'\".,?")}
            return {"intent": "eda"}

    @staticmethod
    def _generate_eda_sql(groq_client, user_query, schemas, history):
        schema_desc = ""
        for t_name, schema in schemas.items():
            schema_desc += f'Table: "{t_name}"\n'
            schema_desc += "\n".join([f"  - {col} ({dtype})" for col, dtype in schema.items()])
            schema_desc += "\n\n"
            
        resp = IntelligentChatAgent._api_call_with_retry(
            client=groq_client,
            messages=IntelligentChatAgent._build_messages(f"""You are a CockroachDB SQL expert. Generate ONLY a single safe SQL query to answer the user's question or fulfill their request.
Available Schemas:
{schema_desc}
Rules:
1. Output ONLY a JSON object: {{"sql": "...", "explanation": "..."}}
2. Generate a SELECT statement for analysis. If the user explicitly asks to add, update, delete, clean, or modify data (e.g., adding missing values or new rows), generate the appropriate INSERT, UPDATE, or DELETE statement.
3. NEVER generate DROP TABLE, ALTER TABLE, or schema-destroying queries.
4. Choose the most appropriate table based on the user's question. If the user asks about a specific file, query its corresponding table.
5. Use NULL checks with IS NULL / IS NOT NULL
6. For distributions: use GROUP BY with COUNT(*)
7. Keep LIMIT ≤ 50 for row-fetching queries
8. Column names with spaces must be quoted with double quotes
9. Do NOT reference the 'embedding' column""", history, user_query),
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=512
        )
        raw = resp.choices[0].message.content.strip()
        
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                return parsed.get("sql", ""), parsed.get("explanation", "")
            except Exception:
                pass
                
        sql_match = re.search(r'(?:SELECT|INSERT|UPDATE|DELETE)[\s\S]+?(?:;|$)', raw, re.IGNORECASE)
        if sql_match:
            return sql_match.group().strip(), "Generated query"
            
        return "", ""

    @staticmethod
    def generate_dynamic_graph_from_csv(user_query, csv_url):
        import os
        from openai import OpenAI
        import re
        
        try:
            groq_client = OpenAI(base_url='https://api.groq.com/openai/v1', api_key=os.environ.get('GROQ_API_KEY'))
            
            prompt = f"""You are a Python data science plotting expert. Write a Python script using pandas, matplotlib, and seaborn to generate a plot for the user's request.
User request: "{user_query}"

Requirements:
1. The dataset URL is available as a global variable `CSV_URL`. Load it exactly like this: `import pandas as pd; df = pd.read_csv(CSV_URL)`
   (CRITICAL: Do NOT define or assign the CSV_URL variable in your code. It is already defined in the environment.)
2. Generate the plot based on the user's request using the loaded `df`.
3. Save the figure EXACTLY to: "/tmp/dynamic_plot.png"
4. Do NOT use plt.show(). Do NOT include any ```python wrappers, just the raw code.
5. If there are column name mismatches, print out the actual columns or just infer them. Assume standard naming.
"""
            resp = IntelligentChatAgent._api_call_with_retry(
            client=groq_client,
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=1000
        )
            code = resp.choices[0].message.content.strip()
            code = re.sub(r'^```[a-z]*\n?', '', code, flags=re.MULTILINE)
            code = re.sub(r'```\s*$', '', code, flags=re.MULTILINE)
            
            try:
                if os.path.exists("/tmp/dynamic_plot.png"):
                    os.remove("/tmp/dynamic_plot.png")
                exec_globals = globals().copy()
                exec_globals['CSV_URL'] = csv_url
                exec(code, exec_globals)
                if os.path.exists("/tmp/dynamic_plot.png"):
                    return True
            except Exception as e:
                print(f"Dynamic plot generation failed: {e}")
                return False
                
            return False
        except Exception as e:
            print(f"Error in generate_dynamic_graph_from_csv: {e}")
            return False

    @staticmethod
    def _check_data_quality(conn, table_name, schema):
        issues = []
        total_rows = 0
        try:
            with conn.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                total_rows = cur.fetchone()[0]
        except Exception:
            return issues, total_rows
        
        numeric_types = {'integer', 'bigint', 'numeric', 'double precision', 'real', 'smallint'}
        
        for col, dtype in schema.items():
            try:
                with conn.cursor() as cur:
                    cur.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NULL')
                    null_count = cur.fetchone()[0]
                    if null_count > 0:
                        issue = {
                            "column": col,
                            "issue": "missing_values",
                            "count": null_count,
                            "pct": round((null_count / total_rows) * 100, 1) if total_rows > 0 else 0
                        }
                        if dtype in numeric_types:
                            cur.execute(f'SELECT AVG(CAST("{col}" AS FLOAT)) FROM "{table_name}" WHERE "{col}" IS NOT NULL')
                            mean_val = cur.fetchone()[0]
                            if mean_val is not None:
                                issue["suggestion"] = "fill_mean_or_median"
                                issue["mean"] = round(float(mean_val), 4)
                        else:
                            cur.execute(f'SELECT "{col}", COUNT(*) as cnt FROM "{table_name}" WHERE "{col}" IS NOT NULL GROUP BY "{col}" ORDER BY cnt DESC LIMIT 1')
                            mode_res = cur.fetchone()
                            if mode_res:
                                issue["suggestion"] = "fill_mode_or_unknown"
                                issue["mode"] = str(mode_res[0])
                        issues.append(issue)
            except Exception as e:
                print(f"Data quality check failed for {col}: {e}")
        return issues, total_rows

    @staticmethod
    def _format_results_with_llm(groq_client, user_query, sql, cols, rows, explanation):
        is_dml = any(sql.strip().upper().startswith(kw) for kw in ['INSERT', 'UPDATE', 'DELETE'])
        if not rows and not cols and not is_dml:
            return "The query returned no results."
        
        data_summary = {
            "query_executed": sql,
            "explanation": explanation,
            "status": "Success (Data Modified)" if is_dml else "Success (Data Retrieved)",
            "columns": cols,
            "rows": rows[:50],
            "total_rows_returned": len(rows)
        }
        
        summary_resp = IntelligentChatAgent._api_call_with_retry(
            client=groq_client,
            messages=[{
                "role": "system",
                "content": """You are a highly intelligent Data Scientist AI presenting database results to a user.
Format the provided data clearly and beautifully using Markdown.
Rules:
1. NEVER invent numbers or facts not present in the data.
2. Use a Markdown table if results have multiple rows/columns.
3. Use **bold** for key statistics.
4. Provide a 1-2 sentence plain-English summary at the top.
5. Proactively analyze the data: point out missing values, interesting anomalies, or key trends if they exist in the result.
6. Do NOT repeat the SQL query to the user.
7. ALWAYS end your response with a "### Recommended Follow-up Questions:" section containing 2-3 highly intelligent, proactive questions the user can ask next to further explore, clean, or model this specific data."""
            }, {
                "role": "user",
                "content": f"Original Question: {user_query}\n\nSQL Results:\n{json.dumps(data_summary, indent=2)}"
            }],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024
        )
        try:
            return summary_resp.choices[0].message.content.strip()
        except Exception as e:
            return f"AI Error formatting results: Could not process response from OpenRouter ({str(e)}). This is usually caused by the payload being too large."

    @staticmethod
    def handle_agent_chat(user_query, db_url, pending_clean_context=None, history=None):
        if history is None: history = []
        print(f">>> [Chat Agent] Query: {user_query}")
        start_time = time.time()
        
        if db_url:
            os.environ["DATABASE_URL"] = db_url

        groq_client = OpenAI(base_url='https://api.groq.com/openai/v1', api_key=os.environ.get('GROQ_API_KEY'))
        
        conn = None
        table_names = []
        schemas = {}
        try:
            conn = VectorEngine.get_db_conn()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name NOT IN ('model_experiments')
                """)
                table_names = [row[0] for row in cur.fetchall()]
        except Exception as e:
            return {"answer": f"⚠️ Could not connect to the database: {e}"}
        
        if not table_names:
            return {"answer": "I don't have any data loaded yet. Please upload a dataset first!"}
        
        for t in table_names:
            schemas[t] = IntelligentChatAgent._get_table_schema(conn, t)
        
        intent_data = IntelligentChatAgent._classify_intent(groq_client, user_query, history)
        intent = intent_data.get("intent", "eda")
        print(f">>> [Chat Agent] Intent: {intent_data}")
        
        if intent == "train":
            target = intent_data.get("target", "").strip()
            if not target:
                return {"answer": "I'd love to train a model! Which column would you like to predict? (e.g. 'Price', 'Category', 'Churn')"}
            return {"intent": "train", "target": target}
        
        if intent == "clean_action" and pending_clean_context:
            confirmed = intent_data.get("confirm", False)
            if not confirmed:
                conn.close()
                return {
                    "answer": "No problem! I'll leave the data as-is. You can clean it yourself and re-upload, or proceed to train the model directly.",
                    "pending_clean": None
                }
            col = pending_clean_context.get("column")
            method = intent_data.get("method", pending_clean_context.get("default_method", "mean"))
            fill_value = None
            if method in ("mean", "avg", "average") and pending_clean_context.get("mean") is not None:
                fill_value = pending_clean_context["mean"]
            elif method in ("mode", "most_frequent") and pending_clean_context.get("mode") is not None:
                fill_value = f"'{pending_clean_context['mode']}'"
            elif method in ("median",) and pending_clean_context.get("mean") is not None:
                fill_value = pending_clean_context["mean"]
            else:
                fill_value = f"'{method}'"
            
            table_to_clean = pending_clean_context.get("table", table_names[0])
            clean_sql = f'UPDATE "{table_to_clean}" SET "{col}" = {fill_value} WHERE "{col}" IS NULL'
            print(f">>> [Clean Agent] Executing: {clean_sql}")
            _, _, err = IntelligentChatAgent._safe_sql_execute(conn, clean_sql)
            conn.close()
            
            if err:
                return {"answer": f"❌ Cleaning failed for **{col}**: `{err}`\n\nYou may need to handle this manually."}
            
            return {
                "answer": f"✅ **Done!** Filled **{pending_clean_context.get('count', '?')} missing values** in column `{col}` using **{method}** in `{table_to_clean}`.\n\nThe data is now clean. Would you like me to check for more issues, or are you ready to **train the model**?",
                "pending_clean": None
            }
        
        max_retries = 3
        err = None
        sql = ""
        explanation = ""
        cols = []
        rows = []
        retry_history = history.copy() if history else []
        
        for attempt in range(max_retries):
            sql, explanation = IntelligentChatAgent._generate_eda_sql(groq_client, user_query, schemas, retry_history)
            if not sql:
                conn.close()
                return {"answer": "I couldn't generate a valid SQL query for that question. Could you rephrase it? For example: *'How many missing values are in the Age column?'* or *'What is the average price?'*"}
            print(f">>> [EDA Agent] Generated SQL (Attempt {attempt+1}): {sql}")
            cols, rows, err = IntelligentChatAgent._safe_sql_execute(conn, sql)
            if not err:
                break
            print(f">>> [EDA Agent] SQL Error: {err}")
            retry_history.append({"role": "assistant", "content": f'{{"sql": "{sql}", "explanation": "{explanation}"}}'})
            retry_history.append({"role": "user", "content": f"The query encountered a SQL syntax or execution error: {err}. Please fix the query syntax (make sure to quote reserved words, match brackets correctly, etc) and try again."})
            
        if err:
            conn.close()
            return {"answer": f"⚠️ I tried to generate a query but encountered an issue: `{err}`\n\nI tried to auto-resolve it but failed. Try rephrasing your question!"}
        
        answer = IntelligentChatAgent._format_results_with_llm(groq_client, user_query, sql, cols, rows, explanation)
        
        used_table = table_names[0]
        for t in table_names:
            if f'"{t}"' in sql or f' {t} ' in sql or f' {t};' in sql or f' {t}\n' in sql:
                used_table = t
                break

        pending_clean = None
        quality_prompt_addition = ""
        try:
            issues, total_rows = IntelligentChatAgent._check_data_quality(conn, used_table, schemas[used_table])
            if issues:
                worst = max(issues, key=lambda x: x["count"])
                col = worst["column"]
                count = worst["count"]
                pct = worst["pct"]
                
                if worst.get("mean") is not None:
                    method_hint = f"fill them with the **mean** ({worst['mean']}) or **median**"
                    default_method = "mean"
                elif worst.get("mode") is not None:
                    method_hint = f"fill them with the **most frequent value** (`{worst['mode']}`) or mark as `Unknown`"
                    default_method = "mode"
                else:
                    method_hint = "drop those rows"
                    default_method = "drop"
                
                quality_prompt_addition = f"\n\n---\n💡 **Data Quality Alert (`{used_table}`):** I noticed `{col}` has **{count} missing values** ({pct}% of {total_rows} rows). Want me to {method_hint}? Just say **yes** and specify the method, or **no** to handle it yourself."
                
                pending_clean = {
                    "table": used_table,
                    "column": col,
                    "count": count,
                    "mean": worst.get("mean"),
                    "mode": worst.get("mode"),
                    "default_method": default_method
                }
        except Exception as e:
            print(f">>> [Quality Check] Failed: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
        
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "answer": answer + quality_prompt_addition,
            "sql_executed": sql,
            "metrics": {"latency_ms": latency_ms},
            "pending_clean": pending_clean
        }
