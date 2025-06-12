<!-- # MNLP-M3: RAG-Enhanced Instruction Tuning Pipeline -->
# RAG STEM MCQA Tutor

A minimal pipeline for retrieval-augmented multiple-choice question answering (MCQA) in STEM domains, developed as part of the CS-552 MNLP course project.

This project combines retrieval and generation to answer STEM MCQs using:

- Chunked corpus construction and FAISS-based indexing
- Dense retrieval with a finetuned [`thenlper/gte-small`](https://huggingface.co/thenlper/gte-small) model
- Instruction-tuned generation using [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base), with both SFT and LoRA variants

---

## Structure

```
.
├── data_collectors/         # Data scraping modules (HTML + Wikipedia)
│   ├── collector_base.py
│   ├── html_collector.py
│   ├── wiki_collector.py
│   └── utils.py
├── encoder/                 # Sentence embedding pipeline
│   ├── data_generator.py
│   └── train_embedding.py
├── generator/               # Instruction tuning pipeline
│   ├── data_generator.py    # Corpus chunking + FAISS + retrieval-based augmentation
│   ├── train_sft.py        
│   └── train_lora.py        
├── corpus_generator.py      
├── run.py                 
├── requirements.txt
└── README.md
```

---

## How-to

```bash
python -m venv m3env
source m3env/bin/activate
pip install -r requirements.txt
```
Training
```bash
python run.py
```
Example
```bash
python mcqa_demo.py
```
---

## Outputs

- Embedding model pushed to: `sucharush/MNLP_M3_document_encoder`
- Generator pushed to: `sucharush/MNLP_M3_rag_model`
- Dataset stored in: `sucharush/MNLP_M3_rag_dataset`

## Results
| Model Variant    | MMLU-Pro (%) | ARC-Challenge (%)|
| ---------------- | --------- | ------------- |
| Base             | 43.19     | 65.61         |
| SFT              | 42.30     | 65.36         |
| SFT + LoRA           | 44.56     | 65.02         |
| SFT + RAG            | 43.25     | 63.82         |
| SFT + LoRA + RAG | **45.52** | **65.70**     |
Evaluated with top-*k*=2 retrieved documents per question.