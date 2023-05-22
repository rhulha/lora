import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, AutoModel
from peft import PeftModel, PeftConfig

load_in_8bit = True
org_model_name = "eachadea/vicuna-13b-1.1"
lora_modelpath="my_lora"
device = "cuda"
debug = False


@torch.inference_mode()
def generate_stream(model, tokenizer, params, device,
                    context_len=2048, stream_interval=2):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            out = model(
                torch.as_tensor([input_ids], device=device), use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            attention_mask = torch.ones(
                1, past_key_values[0][0].shape[-2] + 1, device=device)
            out = model(input_ids=torch.as_tensor([[token]], device=device),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            pos = output.rfind(stop_str, l_prompt)
            if pos != -1:
                output = output[:pos]
                stopped = True
            yield output

        if stopped:
            break

    del past_key_values

peft_config = PeftConfig.from_pretrained(lora_modelpath)

tokenizer = AutoTokenizer.from_pretrained(org_model_name, use_fast=False)
#tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        load_in_8bit=load_in_8bit,
        return_dict=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(
        model,
        lora_modelpath,
        torch_dtype=torch.float16,
        device_map="auto",
)

print( next(model.parameters()).device )

model.to(device)


if debug:
    print(model)

system="A chat between a curious human and an artificial intelligence assistant. " \
           "The assistant gives helpful, detailed, and polite answers to the human's questions."
roles=("Human", "Assistant")

seperator = "###"

while True:
    try:
        message = input("Human: ")
        prompt = system + seperator + roles[0] + ": " + message + seperator
        skip_echo_len = len(prompt) + 1

        print ( prompt )
        
        params = {
            "prompt": prompt,
            "temperature": 0.7,
            "max_new_tokens": 512,
            "stop": seperator
        }

        print(f"{roles[1]}: ", end="", flush=True)

        pre = 0
        for outputs in generate_stream(model, tokenizer, params, device):
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs)
            if now - 1 > pre:
                print(" ".join(outputs[pre:now-1]), end=" ", flush=True)
                pre = now - 1
        print(" ".join(outputs[pre:]), flush=True)

    except EOFError:
        prompt = ""
    if not prompt:
        print("exit...")
        break
