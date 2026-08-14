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
