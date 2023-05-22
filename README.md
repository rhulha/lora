# lora
Train Large Language Models (LLM) using Huggingface, PEFT and LoRA  

I have included a script that sets up most of the things needed if you use lambdalabs.  
It is called: setup_lambdalabs.py  

To use this script you will need to create a .env file  
containing these three entries:

LL_SECRET=my_lambda_labs_secret  
ssh_key_filename=my_path_to_my_private_rsa_key  
training_data_url=https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt  

The LoRA training script is called train_text.py  

Please look at the header of the train_text.py script to adjust settings like:  

model_name = "eachadea/vicuna-13b-1.1"  
load_in_8bit=True  
lora_file_path = "my_lora"  
text_filename='input.txt'  
output_dir='.'  
cutoff_len = 512  
overlap_len = 128  
newline_favor_len = 128  

Very Important: There is currently a problem saving the LoRA model.
User angelovAlex found a great solution here: https://github.com/rhulha/lora/issues/1

The setup_lambdalabs.py will automatically apply this patch.

If you don't use lambdalabs you will have to apply this patch manually.

To use the LoRA model you can take a look at inference.py.

It also uses hard coded values, so if you change model names you will have to adapt his script too.

