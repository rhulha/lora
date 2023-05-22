import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import Dataset, load_dataset
import transformers
from peft import (LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict)

model_name = "eachadea/vicuna-13b-1.1"
load_in_8bit=True
lora_file_path = "my_lora"
text_filename='input.txt'
output_dir='.'
cutoff_len = 512
overlap_len = 128
newline_favor_len = 128

def split_chunks(arr, step):
    for i in range(0, len(arr), step):
        yield arr[i:i + step]

def cut_chunk_for_newline(chunk: str, max_length: int):
    if '\n' not in chunk:
        return chunk
    first_newline = chunk.index('\n')
    if first_newline < max_length:
        chunk = chunk[first_newline + 1:]
    if '\n' not in chunk:
        return chunk
    last_newline = chunk.rindex('\n')
    if len(chunk) - last_newline < max_length:
        chunk = chunk[:last_newline]
    return chunk

def tokenize(prompt):
    result = tokenizer(prompt, truncation=True, max_length=cutoff_len + 1, padding="max_length")
    return {
        "input_ids": result["input_ids"][:-1], # return all elements except the last one.
        "attention_mask": result["attention_mask"][:-1], # return all elements except the last one.
    }

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=load_in_8bit, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

config = LoraConfig(
    r=16, # 32 oob
    lora_alpha=32, # 64 oob
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

if not hasattr(model, 'lm_head') or hasattr(model.lm_head, 'weight'):
    print("prepare_model_for_int8_training...")
    prepare_model_for_int8_training(model)

lora_model = get_peft_model(model, config)

with open(text_filename, 'r', encoding='utf-8') as file:
    raw_text = file.read()

tokens = tokenizer.encode(raw_text)
del raw_text  # be safe on RAM
tokens = list(split_chunks(tokens, cutoff_len - overlap_len))
for i in range(1, len(tokens)):
    tokens[i] = tokens[i - 1][-overlap_len:] + tokens[i]

text_chunks = [tokenizer.decode(x) for x in tokens]
del tokens
if newline_favor_len > 0:
    text_chunks = [cut_chunk_for_newline(x, newline_favor_len) for x in text_chunks]

train_data = Dataset.from_list([tokenize(x) for x in text_chunks])
del text_chunks

trainer = transformers.Trainer(
    model=lora_model, 
    train_dataset=train_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        evaluation_strategy="no",
        logging_steps=1, 
        output_dir=output_dir,
        ddp_find_unused_parameters=None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
lora_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

if torch.__version__ >= "2" and sys.platform != "win32":
    lora_model = torch.compile(lora_model)

trainer.train()
lora_model.save_pretrained(lora_file_path)

