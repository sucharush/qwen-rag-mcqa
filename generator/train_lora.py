import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType

# -------- Config --------
model_name = "sucharush/camel_qwen_sft_small"
dataset_name = ("sucharush/MNLP_M3_rag_dataset", "mcqa_with_doc_2_512")
output_dir = "./qwen_lora_rag_mcqa-final_2_512"
run_name = "qwen_lora_rag_mcqa-final_2_512"

def preprocess(example, tokenizer):
    prompt = example["prompt"].strip()
    response = example["response"].strip()
    separator = "\n\n"
    prompt_part = prompt + separator
    full_text = prompt_part + response

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        padding=False,
        max_length=1536,
        add_special_tokens=True,
    )

    prompt_ids = tokenizer(
        prompt_part,
        truncation=False,
        padding=False,
        add_special_tokens=False
    )["input_ids"]

    full_ids = full_tokens["input_ids"]
    prompt_len = len(prompt_ids) if full_ids[:len(prompt_ids)] == prompt_ids else 0
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {
        "input_ids": full_ids,
        "attention_mask": full_tokens["attention_mask"],
        "labels": labels[:len(full_ids)],
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    dataset = load_dataset(*dataset_name)
    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = split["train"].map(
        lambda ex: preprocess(ex, tokenizer),
        remove_columns=split["train"].column_names,
        desc="Processing training data"
    )
    eval_data = split["test"].map(
        lambda ex: preprocess(ex, tokenizer),
        remove_columns=split["test"].column_names,
        desc="Processing evaluation data"
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(base_model, peft_config)
    model.to(device)
    # model.gradient_checkpointing_enable()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding=True,
        return_tensors="pt"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=3,
        gradient_accumulation_steps=6,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        learning_rate=5e-5,
        eval_strategy="steps",
        eval_steps=400,
        save_strategy="steps",
        save_steps=1200,
        save_total_limit=4,
        logging_steps=400,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    try:
        print("Starting LoRA training...")
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
