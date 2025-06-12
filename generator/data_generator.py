from datasets import load_dataset
from generator.rag_utils import (
    get_tokenizer,
    chunk_document,
    flatten_chunks,
    encode_corpus,
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    retrieve_and_format_sample
)
import os


# -------- Config --------
corpus_repo = "sucharush/stem_corpus"
output_repo = "sucharush/MNLP_M3_rag_dataset"
embedding_model = "sucharush/MNLP_M3_document_encoder"
target_mcqa_repo = "sucharush/rag_sft_mcqa_small"
faiss_path = "faiss_cache/index.faiss"
chunk_size = 512
stride = 32
top_k = 2

print("Loading corpus and chunking...")
tokenizer = get_tokenizer(embedding_model)
raw_corpus = load_dataset(corpus_repo, split="train").select(range(100))
chunked = raw_corpus.map(
    lambda x: chunk_document(x, tokenizer, chunk_size, stride),
    remove_columns=raw_corpus.column_names
)

print("Flattening chunked dataset...")
flattened = chunked.map(flatten_chunks, batched=True, remove_columns=["chunks"])
flattened.push_to_hub("sucharush/stem_corpus_chunked")

# print("Building FAISS index...")
if os.path.exists(faiss_path):
    print("Loading FAISS index from cache...")
    index = load_faiss_index(faiss_path)
else:
    embeddings, encoder = encode_corpus(flattened["text"], embedding_model)
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    save_faiss_index(index, faiss_path)


print("Loading MCQA dataset...")
mcqa = load_dataset(target_mcqa_repo, split="train")

print("Performing document retrieval and augmenting prompts...")
augmented = mcqa.map(lambda x: retrieve_and_format_sample(x, encoder, index, flattened, k=top_k))

print("Pushing augmented dataset to hub...")
augmented.push_to_hub(output_repo, config_name="mcqa_with_doc_2_512")


