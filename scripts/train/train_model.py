from mathmodel.data import load_data
from mathmodel.model import load_model, save_model
from mathmodel.train import train_model

# change the dataset name and model name 
dataset = load_data("your-dataset-name")
model, tokenizer = load_model("your-model-name")

trained_model = train_model(model, dataset)
save_model(trained_model, "math_model/outputs")
