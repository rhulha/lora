import paramiko
import requests
from dotenv import dotenv_values

config = dotenv_values("./.env")
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
    LambdLabsServer = ip

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(LambdLabsServer, username="ubuntu", key_filename="C:\\Users\\Ray\\.ssh\\id_rsa")

sftp = ssh.open_sftp()
sftp.get("./my_lora.zip", './my_lora.zip')
sftp.close()

ssh.close()
