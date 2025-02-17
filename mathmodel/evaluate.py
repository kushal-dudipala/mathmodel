import torch

def evaluate_model(model, tokenizer, prompt):
    """Generates a response from the DeepSeek-Math model."""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs, max_length=10000, pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response
