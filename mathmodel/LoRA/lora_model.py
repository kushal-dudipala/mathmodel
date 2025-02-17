import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_lora_model(
    model_name="deepseek-ai/deepseek-math-7b-instruct",
    offload_folder="/home/hice1/ychauhan9/scratch",
    lora_r=8, lora_alpha=32, lora_dropout=0.05
):
    """Loads the model and applies LoRA adapters."""

    token = os.getenv("HUGGINGFACE_TOKEN")

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

    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)

    # Define LoRA configuration
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    return model, tokenizer
