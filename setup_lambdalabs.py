import requests
import paramiko
from dotenv import dotenv_values

config = dotenv_values("./.env")
api_key = config["LL_SECRET"]
ssh_key_filename = config["ssh_key_filename"]
training_data_url = config["training_data_url"]

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
ssh.connect(LambdLabsServer, username="ubuntu", key_filename=ssh_key_filename)

def run_cmd(cmd):
    print(cmd)
    stdin, stdout, stderr = ssh.exec_command(cmd)
    for line in stdout:
        print(line, end='')

def run_basic_setup():
    run_cmd("sudo rm /var/lib/man-db/auto-update") # https://askubuntu.com/questions/272248/processing-triggers-for-man-db
    run_cmd("sudo apt-get update -y")
    #run_cmd("sudo apt-get upgrade -y")
    run_cmd("sudo apt-get install git-lfs -y")
    run_cmd("python3 -m pip install --upgrade pip")
    run_cmd("pip3 install transformers")
    run_cmd("pip3 install datasets accelerate loralib peft")
    run_cmd("pip3 install sentencepiece")
    run_cmd("pip3 install protobuf==3.20.*")


def setup_LD_LIBRARY_PATH():
    #run_cmd("find /usr/lib/ -name libcudart.so 2>/dev/null") # use this to find it
    run_cmd("export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/")
    run_cmd('echo "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/">>~/.bashrc')

def replace_in_file(file, old_string, new_string):
        with open(file, 'r') as f:
            s = f.read()
            s = s.replace(old_string, new_string)
        with open(file, 'w') as f:
            f.write(s)

def compile_bits_and_bytes(patch_save_pretrained=True):
    run_cmd("git clone https://github.com/TimDettmers/bitsandbytes")
    run_cmd("cd bitsandbytes;pip install -r requirements.txt")
    run_cmd("nvcc --version")

    if patch_save_pretrained:
        # patch_save_pretrained begin
        # there is a problem with saving the model with running out of GPU memory
        # this patch fixes this problem. Credit goes to angelovAlex
        sftp = ssh.open_sftp()
        sftp.get('/home/ubuntu/bitsandbytes/bitsandbytes/nn/modules.py', './modules.py')
        replace_in_file('./modules.py', "self.state.CxB, self.state.tile_indices", "self.state.CxB.cpu(), self.state.tile_indices.cpu()")
        sftp.put('./modules.py', '/home/ubuntu/bitsandbytes/bitsandbytes/nn/modules.py')
        sftp.close()
        # patch_save_pretrained end

    run_cmd("cd bitsandbytes;CUDA_VERSION=118 make cuda11x")
    run_cmd("cd bitsandbytes;sudo python setup.py install")

def upload_training_data():
    run_cmd("wget " + training_data_url)

def upload_train_py():
    sftp = ssh.open_sftp()
    sftp.put('train_text.py', '/home/ubuntu/train_text.py')
    sftp.put('inference.py', '/home/ubuntu/inference.py')
    sftp.close()

run_basic_setup()
setup_LD_LIBRARY_PATH()
compile_bits_and_bytes()
upload_training_data()
upload_train_py()

print(ip)

ssh.close()

