from mathmodel.model import load_model
from mathmodel.evaluate import evaluate_model

# Load model and tokenizer
model, tokenizer = load_model()

# Test prompt
prompt = "Return the answer to 2 + 2 and nothing else."

# Generate response
response = evaluate_model(model, tokenizer, prompt)

print("\n🔹 DeepSeek Response:\n", response)
