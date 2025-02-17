import torch
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, prepare_model_for_kbit_training

def train_model(model, dataset, output_dir="outputs"):
    """Fine-tunes the model with given dataset."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    trainer.train()
    return model
