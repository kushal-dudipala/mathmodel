import os
from datasets import load_dataset

def load_data(dataset_name="default_dataset"):
    """Loads and preprocesses dataset for training."""
    dataset = load_dataset(dataset_name)
    return dataset
