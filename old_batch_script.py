# --------------------------
# 1. Environment Setup
# --------------------------
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    GenerationConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel

# T4-specific settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# scratch dir: /home/hice1/ychauhan9/scratch

# --------------------------
# 2. Model & Tokenizer Loading
# --------------------------
model_name = "deepseek-ai/deepseek-math-7b-instruct"
token = os.getenv("HUGGINGFACE_TOKEN")

# 4-bit config for T4
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # float16 for T4 compatibility
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    token=token,
    offload_folder='/home/hice1/ychauhan9/scratch'
)
# Generate response
prompt = "Return the answer to 2 + 2 and nothing else."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_length=10000,pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n🔹 DeepSeek Response:\n", response)