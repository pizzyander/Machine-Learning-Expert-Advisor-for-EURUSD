import requests

TOKEN = "7846836503:AAHPPkb8pNHsEGL4d13es_nrbf7qWMBtBQQ"
url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

response = requests.get(url)

# Ensure the response is properly printed
print(response.text.encode('utf-8').decode('utf-8'))
