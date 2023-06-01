import requests

url = "http://localhost:5000/predict"

data = {
    "age": 18393,
    "gender": 2,
    "height": 168,
    "weight": 62,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
}

response = requests.post(url, json=data)
print(response.json())