import requests
url = "http://127.0.0.1:5000/predict"
resp = requests.post(url, files={'file':open('indiv.csv', 'rb')})
print(resp.text)