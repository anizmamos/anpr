import requests

with open('test.jpg', 'rb') as f:
    response = requests.post('http://127.0.0.1:4000/anpr', files={'image': f})
    print(response.json())
