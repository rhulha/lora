from dotenv import dotenv_values
import requests
import os

config = dotenv_values(".env")
api_key = config["LL_SECRET"]
response = requests.get('https://cloud.lambdalabs.com/api/v1/instances', auth=(api_key, ''))

response_data = response.json()

response_data = response_data["data"]

for instance in response_data:
    id = instance["id"]
    ip = instance["ip"]
    gpu = instance["instance_type"]["name"]
    jupyter_url = instance["jupyter_url"]
    print(id, ip, gpu, jupyter_url)
    os.system('start ' + jupyter_url)

