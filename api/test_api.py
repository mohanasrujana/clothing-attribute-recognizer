import requests
import json

# Define the API endpoint
url = "http://127.0.0.1:5000/predict"

# Define input dataset and output directory paths
input_dataset_path = "/Users/satyasrujanapilli/Downloads/clothing-attribute-recognizer/dataset/img/2-in-1_Space_Dye_Athletic_Tank/"
output_file_path = "/Users/satyasrujanapilli/Downloads/Results/"

# Prepare request payload in the expected format
data = {
    "inputs": {
        "input_dataset": {"path": input_dataset_path},
        "output_file": {"path": output_file_path}
    },
    "parameters": {
        "threshold": "0.5"
    }
}

# Send the POST request
response = requests.post(url, json=data)

try:
    response_json = response.json()

    # Extract file path directly
    predictions_path = response_json.get("path")
    if predictions_path:
        print("Predictions saved at:", predictions_path)
    else:
        print("Error: Predictions file path is missing in the API response.")
except json.JSONDecodeError:
    print(f"Error {response.status_code}: {response.text}")
except Exception as e:
    print(f"Unexpected error: {e}")