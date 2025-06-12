import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from generator.rag_utils import (
    get_tokenizer,
    chunk_document,
    flatten_chunks,
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
)
import faiss
import torch
import numpy as np

# -------- Config --------
embedding_model = "sucharush/MNLP_M3_document_encoder"
generator_model = "sucharush/camel_qwen_sft_small"
chunked_repo = "sucharush/stem_corpus_chunked"
faiss_path = "./faiss_cache/corpus_faiss.index"
chunk_size = 512
stride = 32
top_k = 2

# -------- Load chunked dataset or build --------
try:
    print("Trying to load chunked dataset from HF...")
    flattened = load_dataset(chunked_repo, split="train")
    print(len(flattened))
except ValueError:
    print("Chunked dataset not found. Rebuilding and uploading...")
    tokenizer = get_tokenizer(embedding_model)
    raw_corpus = load_dataset("sucharush/stem_corpus", split="train")
    chunked = raw_corpus.map(
        lambda x: chunk_document(x, tokenizer, chunk_size, stride),
        remove_columns=raw_corpus.column_names
    )
    flattened = chunked.map(flatten_chunks, batched=True, remove_columns=["chunks"])
    flattened.push_to_hub(chunked_repo)
corpus = flattened["text"]  

# -------- Load or build FAISS index --------
retriever = SentenceTransformer(embedding_model)

if os.path.exists(faiss_path):
    print("Loading FAISS index from cache...")
    index = load_faiss_index(faiss_path)
else:
    print("Encoding and building FAISS index...")
    embeddings = retriever.encode(
        flattened["text"],
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    index = build_faiss_index(embeddings)
    save_faiss_index(index, faiss_path)

QUESTION = (
    "Excited states of the helium atom can be characterized as para- "
    "(antiparallel electron spins) and ortho- (parallel electron spins). "
    "The observation that an ortho- state has lower energy than the corresponding "
    "para- state can be understood in terms of which of the following?"
)

OPTIONS = [
    "A. The Heisenberg uncertainty principle",
    "B. The Pauli exclusion principle",
    "C. The Bohr model of the atom",
    "D. Nuclear hyperfine coupling"
]

# ======== Load FAISS index and corpus ========
print("Retrieving relevant documents...")
query_emb = retriever.encode([QUESTION], normalize_embeddings=True)  # already returns np.array
D, I = index.search(query_emb, k=top_k)
retrieved = [corpus[i] for i in I[0]]


# ======== Show retrieved docs  ========
print("====== Retrieved Documents ======")
for i, doc in enumerate(retrieved, 1):
    print(f"[Doc {i}]\n{doc.strip()}\n")

# ======== Prepare prompts ========
plain_prompt = f"Question: {QUESTION}\nOptions:\n" + "\n".join(OPTIONS) + "\nAnswer:"
rag_prompt = "\n".join(retrieved) + "\n\nOptions:\n" + "\n".join(OPTIONS) + "\nAnswer:"

# ======== Load model ========
tokenizer = AutoTokenizer.from_pretrained(generator_model)
model = AutoModelForCausalLM.from_pretrained(generator_model, device_map="auto", torch_dtype=torch.float16)
model.eval()

# ======== Run RAG generation ========
inputs_rag = tokenizer(rag_prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out_rag = model.generate(**inputs_rag, max_new_tokens=64, do_sample=False, eos_token_id=tokenizer.eos_token_id)
decoded_rag = tokenizer.decode(out_rag[0], skip_special_tokens=True).strip()
ans_rag = decoded_rag.split("Answer:")[-1].strip()

print("Answer:")
print(ans_rag)