import torch
from transformers import TrainingArguments, Trainer
from peft import PeftModel

def train_lora_model(model, dataset, output_dir="outputs/lora"):
    """Fine-tunes the model using LoRA."""

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

    # Save LoRA adapters
    model.save_pretrained(output_dir)
    return model
