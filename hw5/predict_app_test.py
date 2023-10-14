import requests


url = 'http://localhost:9696/predict_app'

customer = {"job": "unknown", "duration": 270, "poutcome": "failure"}

response = requests.post(url, json=customer).json()
print(response)