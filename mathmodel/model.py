import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from mathmodel.utils import get_scratch_dir, get_huggingface_token

SCRATCH_DIR = get_scratch_dir()
HUGGINGFACE_TOKEN = get_huggingface_token()

def load_model(model_name="deepseek-ai/deepseek-math-7b-instruct", offload_folder=SCRATCH_DIR):
    """Loads the DeepSeek-Math model with 4-bit quantization for efficient inference."""

    token = HUGGINGFACE_TOKEN

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # Optimized for T4
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        token=token,
        offload_folder=offload_folder
    )

    return model, tokenizer
