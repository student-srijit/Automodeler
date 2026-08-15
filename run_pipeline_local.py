from dotenv import load_dotenv
load_dotenv()

import lambda_function
import json

print('Starting AutoML pipeline directly on train.csv and test.csv...')
response = lambda_function.handle_automl_train('Course', '', 'automodeler-uploads-rimo-1234', 'train.csv')

if 'error' in response:
    print('ERROR:', response['error'])
else:
    print('\n=== GENERATED CODE ===')
    print(response['generated_code'])
    print('\n=== EXECUTION LOGS ===')
    print(response['execution_logs'])
    print('\n=== FINAL METRIC ===')
    print(response['final_metric'])
