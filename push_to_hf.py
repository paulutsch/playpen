from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

base_model = "meta-llama/Meta-Llama-3-8B-Instruct"  
checkpoint_dir = "playpen/models/sft+lora/llama3-8b-final/checkpoint-3690"
repo_id = "pm-25/llama3-8b-sft"

with open("key.json", "r") as f:
    keys = json.load(f)

HF_TOKEN_W = keys["huggingface"]["api_key"]
login(HF_TOKEN_W)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype="auto"
)

# load LoRA adapters
model = PeftModel.from_pretrained(model, checkpoint_dir)

# merge adapters to the model
print("Merging LoRA adapters...")
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model)

print(f"Pushing model to {repo_id}...")
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
print("Done: merged model is uploaded!")
