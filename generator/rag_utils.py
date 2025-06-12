import numpy as np
import faiss
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import os

def save_faiss_index(index, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_faiss_index(path: str):
    return faiss.read_index(path)


def get_tokenizer(name="sucharush/MNLP_M3_document_encoder"):
    return AutoTokenizer.from_pretrained(name)


def chunk_document(example, tokenizer, chunk_size=512, stride=32):
    tokens = tokenizer(example["text"], truncation=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), chunk_size - stride):
        chunk_ids = tokens[i: i + chunk_size]
        if len(chunk_ids) < stride:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        chunks.append({
            "text": chunk_text,
            "source": example["source"]
        })
    return {"chunks": chunks}


def flatten_chunks(batch):
    # batch["chunks"] is a list of lists of dicts (already batched)
    # We need to return a dict of lists
    all_chunks = []
    for chunk_list in batch["chunks"]:
        all_chunks.extend(chunk_list)  # flatten each sample's chunk list
    return {key: [d[key] for d in all_chunks] for key in all_chunks[0]}



def encode_corpus(text_list, model_name, batch_size=64):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings, model


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve_and_format_sample(sample, model, index, corpus, k=2):
    query = sample["prompt"]
    query_emb = model.encode([query], convert_to_numpy=True)
    query_emb = query_emb / np.linalg.norm(query_emb)

    D, I = index.search(query_emb, k)
    docs = [corpus[int(i)]["text"] for i in I[0]]

    context = "\nRelavent Documents:\n"
    context += "\n\n".join([f"Document {i}:::\n{doc}" for i, doc in enumerate(docs)])
    context += "\n\n" + query

    return {
        "prompt": context,
        "response": sample["response"]
    }