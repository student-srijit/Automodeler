import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file before importing lambda_function
load_dotenv()

import lambda_function

app = Flask(__name__)
CORS(app)

@app.route('/api', methods=['POST', 'OPTIONS'])
def api():
    if request.method == 'OPTIONS':
        return '', 200
    
    # Mock the API Gateway event format that the lambda handler expects
    payload = request.get_json(silent=True) or {}
    
    # Intercept 'columns' action for the UI
    if payload.get('action') == 'columns':
        filename = payload.get('filename')
        try:
            import pandas as pd
            import boto3
            # Try local first, then fallback to s3 if running locally with boto3 configured
            try:
                df = pd.read_csv(filename, nrows=0)
            except:
                s3 = boto3.client('s3')
                obj = s3.get_object(Bucket=payload.get('bucket'), Key=filename)
                df = pd.read_csv(obj['Body'], nrows=0)
            return jsonify({'columns': list(df.columns)}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    event = {
        "httpMethod": "POST",
        "body": request.get_data(as_text=True)
    }
    
    # Call the lambda handler directly
    result = lambda_function.lambda_handler(event, None)
    
    # Parse the response back to Flask
    body = json.loads(result.get('body', '{}'))
    status = result.get('statusCode', 200)
    
    return jsonify(body), status

if __name__ == '__main__':
    print("Starting local test server on http://localhost:8000/api")
    print("Paste this URL into your index.html sidebar to test!")
    app.run(port=8000)
