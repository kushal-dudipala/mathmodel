from mathmodel.LoRA.lora_model import load_lora_model
from mathmodel.evaluate import evaluate_model

# Load LoRA model
model, tokenizer = load_lora_model()

# Test prompt
prompt = "Solve the equation: 3x + 7 = 10."

# Generate response
response = evaluate_model(model, tokenizer, prompt)

print("\n🔹 LoRA Model Response:\n", response)
