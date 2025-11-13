# Load model directly
import json
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")



with open("data/prompt.txt", "r") as f:
    prompt = f.read()


with open("./data/text/context_Attrazioni_per_Regione_Molise_103.txt", "r") as f:
    context = f.read()

with open("./data/dataset_info.jsonl", 'r') as f:
    info = [json.loads(line) for line in f]
line = 103
caption = info[line - 1]['caption']

in_prompt = prompt.format(context=context, caption=caption)

messages = [
    {"role": "user", "content": "Who are you?"},
]

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
