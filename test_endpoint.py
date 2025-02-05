import requests
import random

url = "http://127.0.0.1:8000/predict"

# Generate 50 time steps with 14 random float values
data = {
    "features": [[1.2, 3.4, 5.6, 7.8]]
}


headers = {"Content-Type": "application/json"}
response = requests.post(url, json=data, headers=headers)

print(response.status_code, response.json())
