import json
import base64
import uuid
import os
import boto3
import traceback

# AWS Lambda filesystem is strictly read-only except for /tmp.
# We must redirect HOME to /tmp to prevent Errno 30 crashes for standard tools.
os.environ['HOME'] = '/tmp'

from core.utils import s3, detect_modality, stream_status, publish_model_to_s3, log_experiment_to_cockroach
from core.auto_ml import AutoMLAgent
from core.chat_agent import IntelligentChatAgent
from pipeline import AutoModelerPipeline

def lambda_handler(event, context):
    print("Received event:", json.dumps(event)[:200])
    
    # ROUTE 1: API Gateway (Chat Mode)
    if 'httpMethod' in event or 'requestContext' in event:
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
        }
        
        # Support both REST API (v1) and HTTP API (v2) payload formats
        method = event.get('httpMethod') or event.get('requestContext', {}).get('http', {}).get('method')
        
        if method == 'OPTIONS':
            return {'statusCode': 200, 'headers': headers, 'body': ''}
            
        try:
            raw_body = event.get('body') or '{}'
            if event.get('isBase64Encoded'):
                raw_body = base64.b64decode(raw_body).decode('utf-8')
            body = json.loads(raw_body)
            action = body.get('action', 'chat')
            
            if action == 'get_upload_url':
                filename = body.get('filename')
                bucket = body.get('bucket')
                file_hash = body.get('hash')
                if not filename or not bucket:
                    return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing filename or bucket'})}
                try:
                    if file_hash:
                        try:
                            meta = s3.head_object(Bucket=bucket, Key=filename)
                            if meta.get('Metadata', {}).get('sha256') == file_hash:
                                return {'statusCode': 200, 'headers': headers, 'body': json.dumps({'status': 'exists'})}
                        except Exception:
                            pass

                    params = {
                        'Bucket': bucket,
                        'Key': filename,
                        'ContentType': 'application/octet-stream'
                    }
                    if file_hash:
                        params['Metadata'] = {'sha256': file_hash}
                        
                    url = s3.generate_presigned_url(
                        ClientMethod='put_object',
                        Params=params,
                        ExpiresIn=3600
                    )
                    return {
                        'statusCode': 200,
                        'headers': headers,
                        'body': json.dumps({'url': url})
                    }
                except Exception as e:
                    return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': str(e)})}
            
            if action == 'get_eda_graphs':
                filename = body.get('filename')
                bucket = body.get('bucket')
                if not bucket:
                    return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing bucket'})}
                try:
                    if not filename:
                        objs = s3.list_objects_v2(Bucket=bucket)
                        csvs = [o for o in objs.get('Contents', []) if o['Key'].endswith('.csv') or o['Key'].endswith('.tsv')]
                        if not csvs:
                            return {'statusCode': 200, 'headers': headers, 'body': json.dumps({'graphs': []})}
                        csvs.sort(key=lambda x: x['LastModified'], reverse=True)
                        filename = csvs[0]['Key']

                    prefix = f"eda-output/{filename}/"
                    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                    graphs = []
                    for o in objs.get('Contents', []):
                        if o['Key'].endswith('.png'):
                            url = s3.generate_presigned_url(
                                ClientMethod='get_object',
                                Params={'Bucket': bucket, 'Key': o['Key']},
                                ExpiresIn=3600
                            )
                            # Extract title from filename
                            title = os.path.basename(o['Key']).replace('.png', '').replace('_', ' ').title()
                            graphs.append({'title': title, 'url': url})
                    return {
                        'statusCode': 200,
                        'headers': headers,
                        'body': json.dumps({'graphs': graphs})
                    }
                except Exception as e:
                    return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': str(e)})}

            if action == 'upload':
                filename = body.get('filename')
                content_b64 = body.get('content')
                bucket = body.get('bucket')
                
                if not filename or not content_b64 or not bucket:
                    return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing filename, content, or bucket for upload'})}
                
                file_bytes = base64.b64decode(content_b64)
                s3.put_object(Bucket=bucket, Key=filename, Body=file_bytes)

                modality = detect_modality(filename)
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'status': 'success',
                        'message': f'Uploaded {filename} to {bucket}.',
                        'detected_modality': modality
                    })
                }
                
            if action == 'train':
                target = body.get('target', '')
                bucket = body.get('bucket', '')
                if not target or not bucket:
                    return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing target column or bucket'})}
                
                is_async_job = body.get('is_async_job', False)
                job_id = body.get('job_id')

                # If this is the initial request from API Gateway, spawn the background task immediately!
                if not is_async_job:
                    job_id = str(uuid.uuid4())
                    lambda_client = boto3.client('lambda')
                    # Build payload for background invocation
                    async_payload = event.copy()
                    
                    # Update body to include async flags
                    body_dict = json.loads(async_payload.get('body', '{}'))
                    body_dict['is_async_job'] = True
                    body_dict['job_id'] = job_id
                    async_payload['body'] = json.dumps(body_dict)
                    async_payload['isBase64Encoded'] = False # CRITICAL BUGFIX
                    
                    try:
                        lambda_client.invoke(
                            FunctionName=os.environ.get('AWS_LAMBDA_FUNCTION_NAME', 'roy-lambda'),
                            InvocationType='Event',
                            Payload=json.dumps(async_payload)
                        )
                        print(f">>> Spawned background job {job_id}")
                        return {
                            'statusCode': 202,
                            'headers': headers,
                            'body': json.dumps({'status': 'processing', 'job_id': job_id})
                        }
                    except Exception as e:
                        return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': f"Failed to spawn background task: {e}"})}

                # --- This runs ONLY in the background task (is_async_job == True) ---
                print(f">>> Running background job {job_id} for target {target}")
                
                try:
                    try:
                        state_response = s3.get_object(Bucket=bucket, Key="agent_memory_state.txt")
                        db_url = state_response['Body'].read().decode('utf-8').strip()
                    except Exception as e:
                        # Upload error to S3 so frontend polling picks it up
                        err_resp = {'error': 'Could not find database state. Have you uploaded a dataset yet?'}
                        s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}.json", Body=json.dumps(err_resp))
                        return {'statusCode': 400, 'body': ''}
                    
                    # Smart Data Router: Find the CSV that actually contains the target column
                    filename = None
                    try:
                        import pandas as pd
                        import io
                        objs = s3.list_objects_v2(Bucket=bucket)
                        for o in objs.get('Contents', []):
                            if o['Key'].endswith('.csv') and not o['Key'].startswith('jobs/'):
                                try:
                                    obj = s3.get_object(Bucket=bucket, Key=o['Key'], Range='bytes=0-4096')
                                    df_head = pd.read_csv(io.BytesIO(obj['Body'].read()), nrows=0)
                                    if target in df_head.columns:
                                        filename = o['Key']
                                        print(f">>> Smart Router: Selected {filename} for target {target}")
                                        break
                                except:
                                    pass
                    except Exception as e:
                        print(f"Smart Router failed: {e}")
                        
                    if not filename:
                        err_resp = {'error': f"Target column '{target}' not found in any uploaded CSV files."}
                        s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}.json", Body=json.dumps(err_resp))
                        return {'statusCode': 400, 'body': ''}
                        
                    response = AutoMLAgent.handle_automl_train(target, db_url, bucket, filename, job_id=job_id)
                    
                    # ─── S3 Model Registry ───────────────────────────────────
                    stream_status(job_id, bucket, "Publishing trained model to S3 Model Registry...")
                    s3_model_url = publish_model_to_s3(bucket, target)
                    if s3_model_url:
                        response['s3_model_url'] = s3_model_url
                        response['model_download_note'] = f"Model saved to {s3_model_url}"
                    
                    # ─── CockroachDB Experiment Tracker ──────────────────────
                    stream_status(job_id, bucket, "Logging experiment metadata to CockroachDB...")
                    metric_val = response.get('final_metric', 'N/A')
                    reasoning_summary = response.get('nemotron_reasoning', '')
                    rounds = response.get('rounds_taken', 1)
                    try:
                        from re import findall as re_findall
                        nums = re_findall(r'[\d]*\.?[\d]+', str(metric_val))
                        numeric_metric = float(nums[0]) if nums else None
                    except Exception:
                        numeric_metric = None
                    log_experiment_to_cockroach(target, metric_val, numeric_metric, rounds, s3_model_url, reasoning_summary)
                    response['experiment_tracked'] = True
                    
                    # Save the final result to S3 for the frontend to poll!
                    try:
                        s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}.json", Body=json.dumps(response))
                        print(f">>> Successfully saved job {job_id} to S3!")
                    except Exception as e:
                        print(f">>> Failed to save job {job_id} to S3: {e}")
                        
                    return {'statusCode': 200, 'body': 'Background task complete'}
                except Exception as e:
                    print(f">>> Background task CRASHED: {e}")
                    traceback.print_exc()
                    err_resp = {'error': f"Background task failed abruptly: {e}"}
                    s3.put_object(Bucket=bucket, Key=f"jobs/{job_id}.json", Body=json.dumps(err_resp))
                    return {'statusCode': 500, 'body': str(e)}
                
            if action == 'check_job':
                job_id = body.get('job_id')
                bucket = body.get('bucket')
                if not job_id or not bucket:
                    return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing job_id or bucket'})}
                
                try:
                    obj = s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}.json")
                    result_data = json.loads(obj['Body'].read().decode('utf-8'))
                    
                    # Delete the job file to clean up space
                    try:
                        s3.delete_object(Bucket=bucket, Key=f"jobs/{job_id}.json")
                    except Exception:
                        pass
                        
                    return {
                        'statusCode': 200,
                        'headers': headers,
                        'body': json.dumps(result_data)
                    }
                except Exception as e:
                    # If file doesn't exist yet, it's still processing!
                    status_message = "Agent reasoning in progress..."
                    try:
                        status_obj = s3.get_object(Bucket=bucket, Key=f"jobs/{job_id}_status.json")
                        status_data = json.loads(status_obj['Body'].read().decode('utf-8'))
                        status_message = status_data.get('message', status_message)
                    except:
                        pass
                        
                    return {
                        'statusCode': 200,
                        'headers': headers,
                        'body': json.dumps({'status': 'processing', 'message': status_message})
                    }
                
            # action == 'chat'
            user_query = body.get('query', '')
            bucket = body.get('bucket', '')
            pending_clean_context = body.get('pending_clean', None)
            history = body.get('history', [])
            
            if not user_query or not bucket:
                return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Missing query or bucket'})}
                
            try:
                # Fetch DB URL from S3 state
                state_response = s3.get_object(Bucket=bucket, Key="agent_memory_state.txt")
                db_url = state_response['Body'].read().decode('utf-8').strip()
            except Exception as e:
                return {'statusCode': 400, 'headers': headers, 'body': json.dumps({'error': 'Could not find database state. Have you uploaded a dataset yet?'})}
                
            response = IntelligentChatAgent.handle_agent_chat(user_query, db_url, pending_clean_context=pending_clean_context, history=history)
            
            # Fetch EDA graphs and inject into chat response
            if response.get('intent') == 'view_graphs':
                try:
                    objs = s3.list_objects_v2(Bucket=bucket)
                    csvs = [o for o in objs.get('Contents', []) if o['Key'].endswith('.csv') or o['Key'].endswith('.tsv')]
                    if csvs:
                        csvs.sort(key=lambda x: x['LastModified'], reverse=True)
                        filename = csvs[0]['Key']
                        prefix = f"eda-output/{filename}/"
                        eda_objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                        all_graphs = []
                        for o in eda_objs.get('Contents', []):
                            if o['Key'].endswith('.png'):
                                url = s3.generate_presigned_url(
                                    ClientMethod='get_object',
                                    Params={'Bucket': bucket, 'Key': o['Key']},
                                    ExpiresIn=3600
                                )
                                title = os.path.basename(o['Key']).replace('.png', '').replace('_', ' ').title()
                                all_graphs.append({'title': title, 'url': url})
                                
                        # Intelligently filter graphs based on user query
                        filtered_graphs = []
                        try:
                            from openai import OpenAI
                            groq_client = OpenAI(api_key=os.environ.get('GROQ_API_KEY'), base_url="https://api.groq.com/openai/v1")
                            graph_titles = [g['title'] for g in all_graphs]
                            
                            prompt = f"""User query: "{user_query}"
Available graphs: {json.dumps(graph_titles)}

Instructions:
1. Analyze if the user is asking for a specific graph (e.g., "heatmap for survived vs fare", "violin plot for age") OR a general overview (e.g., "show me EDA plots", "visualizations").
2. If specific: Find the best matching graph titles. For example, if they ask for a heatmap, "Correlation Heatmap" is the correct match. Do not include unrelated graphs. If no graph fits, use an empty list.
3. If general: Select 3-5 of the most important graphs (e.g., Correlation Heatmap, and key distributions).
4. Respond EXACTLY with this JSON format and nothing else:
{{"is_specific": true, "titles": ["Title1"]}}
"""
                            
                            llm_resp = groq_client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model="groq/compound",
                                temperature=0.0,
                                response_format={"type": "json_object"}
                            )
                            raw = llm_resp.choices[0].message.content.strip()
                            data_llm = json.loads(raw)
                            chosen_titles = data_llm.get("titles", [])
                            is_specific = data_llm.get("is_specific", False)
                            
                            filtered_graphs = [g for g in all_graphs if g['title'] in chosen_titles]
                            
                            if is_specific and not filtered_graphs:
                                print(f"Attempting dynamic graph generation for: {user_query}")
                                try:
                                    key_obj = s3.get_object(Bucket=bucket, Key="agent_filename.txt")
                                    original_csv_key = key_obj['Body'].read().decode('utf-8').strip()
                                    csv_url = s3.generate_presigned_url(ClientMethod='get_object', Params={'Bucket': bucket, 'Key': original_csv_key}, ExpiresIn=3600)
                                    
                                    success = IntelligentChatAgent.generate_dynamic_graph_from_csv(user_query, csv_url)
                                    if success:
                                        import uuid
                                        dyn_key = f"dynamic-eda/{uuid.uuid4().hex}.png"
                                        s3.upload_file("/tmp/dynamic_plot.png", bucket, dyn_key)
                                        url = s3.generate_presigned_url(ClientMethod='get_object', Params={'Bucket': bucket, 'Key': dyn_key}, ExpiresIn=3600)
                                        filtered_graphs = [{'title': f"Dynamic: {user_query.title()}", 'url': url}]
                                        response['answer'] = "I dynamically generated this custom plot for you based on the data!"
                                    else:
                                        response['answer'] = "I couldn't find that specific plot in the pre-generated EDA graphs, and dynamic generation failed."
                                except Exception as e:
                                    print(f"Dynamic graph prep failed: {e}")
                                    response['answer'] = "I couldn't find that specific plot in the pre-generated EDA graphs, and dynamic generation failed."
                            elif not is_specific and not filtered_graphs:
                                filtered_graphs = all_graphs[:5]
                                
                        except Exception as e:
                            print(f"LLM filtering failed: {e}")
                            filtered_graphs = all_graphs[:5]
                            
                        response['graphs'] = filtered_graphs
                except Exception as e:
                    print(f"Error fetching inline graphs: {e}")
                    pass
            
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps(response)
            }
        except Exception as e:
            traceback.print_exc()
            return {'statusCode': 500, 'headers': headers, 'body': json.dumps({'error': str(e)})}

    # ROUTE 2: S3 Upload (ETL & Provisioning Mode)
    elif 'Records' in event and 's3' in event['Records'][0]:
        return AutoModelerPipeline.run_s3_etl(event)