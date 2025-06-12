import os
from huggingface_hub import list_datasets
import subprocess


def hf_dataset_exists(repo_id: str) -> bool:
    try:
        datasets = list_datasets(author=repo_id.split("/")[0])
        return any(d.id == repo_id for d in datasets)
    except Exception as e:
        print(f"[Warning] Cannot access HF Hub: {e}")
        return False


def run_script(path: str):
    print(f"\n>>> Running: {path}")
    ret = subprocess.run(["python", path])
    if ret.returncode != 0:
        raise RuntimeError(f"Script failed: {path}")


def main():
    dataset_repo = "sucharush/MNLP_M3_rag_dataset"
    corpus_repo = "sucharush/stem_corpus"

    print(f"Checking HF dataset: {dataset_repo} ...")
    if not (hf_dataset_exists(dataset_repo) and hf_dataset_exists(corpus_repo)):
        print("Datasets not found on HF. Building everything locally...")

        # Step 1: Generate dataset
        run_script("corpus_generator.py")
        run_script("encoder/data_generator.py")
        run_script("generator/data_generator.py")

    else:
        print("HF dataset exists. Skipping generation.")

    # Step 2: Train embedding model
    run_script("encoder/train_embedding.py")

    # Step 3: Train main model (SFT + LoRA)
    run_script("generator/train_sft.py")
    run_script("generator/train_lora.py")



if __name__ == "__main__":
    main()
