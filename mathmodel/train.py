import torch
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, prepare_model_for_kbit_training

def train_model(model, dataset, output_dir="outputs"):
    """Fine-tunes the model with given dataset."""
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_steps=200,
        per_device_train_batch_size=1,      # Critical for T4
        gradient_accumulation_steps=8,   # Compensate for small batch
        learning_rate=1e-5,              # Lower rate for stability
        num_train_epochs=2,              # Fewer epochs
        fp16=True,
        gradient_checkpointing=True,     # Save memory
        logging_steps=5,
        optim="paged_adamw_8bit",        # Better for T4
        save_strategy="epoch",
        report_to="none"
    )
        
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    trainer.train()
    return model
