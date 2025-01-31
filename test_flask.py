import requests
url = 'http://localhost:5002/predict'
data = {
    'review_text': 'good movie'
}
response = requests.post(url, json=data)
print(response.json())