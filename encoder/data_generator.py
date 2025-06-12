from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict

# ---- Dataset sources and structure mapping ----
datasets_info = {
    "vblagoje/PubMedQA_instruction": ("instruction", "context"),
    "soufyane/DATA_SCIENCE_QA": ("Question", "Answer"),
    "hanzla/datascience-instruct": ("user", "assistant"),
    "STEM-AI-mtl/Electrical-engineering": ("input", "output"),
    "dkoterwa/camel_ai_maths_instruction_dataset": ("instruction", "response"),
}

def process_dataset(hf_path, q_col, a_col, max_samples=None):
    ds = load_dataset(hf_path, split="train")
    ds = ds.filter(lambda x: x[q_col] and x[a_col])
    if max_samples:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds.map(
        lambda x: {
            "query": f"{x[q_col].strip()}",
            "positive": f"{x[a_col].strip()}",
            "source": hf_path 
        },
        remove_columns=ds.column_names
    )

# ---- Process all datasets ----
all_parts = []
for name, (q, a) in datasets_info.items():
    max_samples = 50000 if name == "vblagoje/PubMedQA_instruction" else None
    processed = process_dataset(name, q, a, max_samples=max_samples)
    all_parts.append(processed)

combined = concatenate_datasets(all_parts)

# ---- Shuffle and cast to DatasetDict if needed
combined = combined.shuffle(seed=42)
dataset_dict = DatasetDict({"train": combined})

# ---- Push to Hugging Face ----
dataset_dict.push_to_hub("sucharush/MNLP_M3_rag_dataset", config_name="embedding_data")