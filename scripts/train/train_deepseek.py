from mathmodel.data import load_data
from mathmodel.model import load_model, save_model
from mathmodel.train import train_model

# Specify dataset and model
DATASET_NAME = "your-dataset-name"
MODEL_NAME = "deepseek-ai/deepseek-math-7b-instruct"

# Load dataset
dataset = load_data(DATASET_NAME)

# Load DeepSeek R1 Math model
model, tokenizer = load_model(MODEL_NAME)

# Train the model
trained_model = train_model(model, dataset)

# Save the trained model
save_model(trained_model, "math_model/outputs")

print("Training complete. Model saved to math_model/outputs/")