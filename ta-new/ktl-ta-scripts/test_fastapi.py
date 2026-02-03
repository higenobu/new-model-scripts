import requests
import json

# FastAPI endpoint URL
url = "http://localhost:8000/analyze"

# Data to send in the POST request
payload = {
    "texts": ["This is fantastic.", "I feel scared."]
}

# Send POST request
response = requests.post(url, json=payload)

# Check if the request was successful
if response.status_code == 200:
    # Parse and print the JSON response
    result = response.json()
    print("Response from API:")
    print(json.dumps(result, indent=4, ensure_ascii=False))  # Pretty print with Unicode support
else:
    # Print error message
    print(f"Error: {response.status_code} - {response.text}")
