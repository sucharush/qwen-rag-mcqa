import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)

# -------- Config --------
model_name = "Qwen/Qwen3-0.6B-Base"
dataset_name = ("sucharush/MNLP_M3_rag_dataset", "camel_subset")
run_name = "camel_qwen_sft_small"
output_dir = "./camel_qwen_sft_small"

def load_data():
    dataset = load_dataset(*dataset_name)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    return split_dataset["train"], split_dataset["test"]

def load_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return tokenizer, model

def preprocess_fn(tokenizer):
    def preprocess(example):
        prompt = example["prompt"].strip()
        response = example["response"].strip()
        separator = "\n\n"
        full_text = prompt + separator + response

        prompt_tokens = tokenizer(
            prompt + separator,
            truncation=False,
            padding=False,
            add_special_tokens=False
        )

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            padding=False,
            max_length=1024,
            add_special_tokens=True
        )

        prompt_len = len(prompt_tokens["input_ids"])
        labels = [-100] * prompt_len + full_tokens["input_ids"][prompt_len:]
        if len(labels) != len(full_tokens["input_ids"]):
            labels = labels[:len(full_tokens["input_ids"])]

        return {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels
        }
    return preprocess

@dataclass
class DataCollatorForCausalLMWithLossMask:
    tokenizer: Any
    padding: bool = True

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [item["labels"] for item in batch] if "labels" in batch[0] else None
        batch_without_labels = [
            {k: v for k, v in item.items() if k != "labels"} for item in batch
        ]
        padded = self.tokenizer.pad(
            batch_without_labels,
            padding=self.padding,
            return_tensors="pt"
        )
        if labels is not None:
            max_length = padded["input_ids"].shape[1]
            padded_labels = []
            for label_seq in labels:
                if len(label_seq) < max_length:
                    padded_label = label_seq + [-100] * (max_length - len(label_seq))
                else:
                    padded_label = label_seq[:max_length]
                padded_labels.append(padded_label)
            padded["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return padded

def main():
    train_data, eval_data = load_data()
    tokenizer, model = load_tokenizer_and_model()
    preprocess = preprocess_fn(tokenizer)

    train_tokenized = train_data.map(
        preprocess,
        remove_columns=train_data.column_names,
        desc="Tokenizing training data"
    )
    eval_tokenized = eval_data.map(
        preprocess,
        remove_columns=eval_data.column_names,
        desc="Tokenizing evaluation data"
    )

    collator = DataCollatorForCausalLMWithLossMask(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=6,
        learning_rate=2e-5,
        num_train_epochs=1,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=400,
        eval_strategy="steps",
        eval_steps=400,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        report_to="none",
        run_name=run_name,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=collator,
    )

    try:
        print("Starting SFT...")
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print("Training completed successfully!")
        # trainer.push_to_hub(f"sucharush/{run_name}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()