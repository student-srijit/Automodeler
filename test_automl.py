import requests
import json
import time

url = 'http://localhost:8000/api'
payload = {
    'action': 'train',
    'bucket': 'automodeler-uploads-rimo-1234',
    'target': 'Course'
}

print('Sending request to AutoML Agent to predict: Course...')
start = time.time()
try:
    response = requests.post(url, json=payload)
    print(f'Response Time: {time.time() - start:.2f}s')
    
    data = response.json()
    if 'error' in data:
        print('ERROR:', data['error'])
    else:
        print('\n=== Generated Code ===')
        print(data['generated_code'])
        print('\n=== Execution Logs ===')
        print(data['execution_logs'])
        print('\n=== Final Metric ===')
        print(data['final_metric'])
        print('\nLatency:', data['metrics']['latency_ms'], 'ms')
        print('\nSUCCESS! The pipeline works perfectly.')
except Exception as e:
    print('Request failed:', e)
