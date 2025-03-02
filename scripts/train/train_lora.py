from mathmodel.data import load_data
from mathmodel.LoRA.lora_model import load_lora_model
from mathmodel.LoRA.lora_train import train_lora_model

# Load dataset
dataset = load_data("your-dataset-name")

# Load model with LoRA
model, tokenizer = load_lora_model()

# Train the model
trained_model = train_lora_model(model, dataset)

print("LoRA training complete. Model saved.")
