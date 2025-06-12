from pathlib import Path
from data_collectors.wiki_collector import WikiCollector
from data_collectors.html_collector import ShervineHTMLCollector
from data_collectors.utils import (
    math_topics_by_module,
    phy_topics_by_module,
    cs_topics_by_module,
    chem_topics_by_module,
    ds_topics_by_module,
    bio_med_topics_by_module,
    eng_topics_by_module,
    urls,
)
import wikipediaapi
from datasets import Dataset, load_dataset
import random, json

######################################################################
######################## Wikipedia entries ###########################
######################################################################
wiki = wikipediaapi.Wikipedia(
    language="en", user_agent="MNLP-Project-RAG/1.0 (A. Cheng)"
)
from huggingface_hub import login

login(token="[hf token here]") 

topic_groups_nested = {
    "math": math_topics_by_module,
    "cs": cs_topics_by_module,
    "ds": ds_topics_by_module,
    "phy": phy_topics_by_module,
    "eng": eng_topics_by_module,
    "chem": chem_topics_by_module,
    "biomed": bio_med_topics_by_module,
}


# Flatten all nested topic lists automatically
def flatten_topics(nested):
    return [topic for sublist in nested.values() for topic in sublist]


# Collect all domain → topic list pairs from nested source
topic_groups = {
    domain: flatten_topics(subgroups)
    for domain, subgroups in topic_groups_nested.items()
}

base_output_dir = Path("wiki_outputs")
base_output_dir.mkdir(exist_ok=True)

output_files = []

for domain, topics in topic_groups.items():
    output_file = base_output_dir / f"{domain}_wiki.jsonl"
    output_files.append(str(output_file))

    print(f"Running domain: {domain}, {len(topics)} topics")

    collector = WikiCollector(
        topics=topics,
        wiki=wiki,
        output_file=output_file,
        # overwrite=False  # skip existing
    )
    try:
        collector.run()
    except Exception as e:
        print(f"Error in domain {domain}: {e}")

print("Finished. Output files:")
for f in output_files:
    print(f)
    
collector = ShervineHTMLCollector(urls = urls, output_file="shervine_cheatsheets_cleaned.jsonl")
collector.run()

######################################################################
######################## Camel AI dataset ############################
######################################################################
random.seed(42)
# --- Config ---
dataset_paths = [
    ("dim/camel_ai_physics", "camel-ai/physics"),
    ("dim/camel_ai_chemistry", "camel-ai/chemistry"),
    ("dim/camel_ai_biology", "camel-ai/biology"),
    ("heya5/camel-ai-math", "camel-ai/math")
]

question_key = "message_1"
answer_key = "message_2"
output_dir = Path("camel_processed_outputs")
output_dir.mkdir(exist_ok=True)

all_data = []

for path, source_tag in dataset_paths:
    ds = load_dataset(path, split="train")
    for ex in ds:
        q = ex.get(question_key, "").strip()
        a = ex.get(answer_key, "").strip()
        if q and a:
            all_data.append({
                "prompt": q,
                "response": a,
                "source": source_tag
            })

print(f"Total camel QA pairs loaded: {len(all_data)}")

# --- Shuffle and 50/50 split ---
random.shuffle(all_data)
half = len(all_data) // 2
# train_data = all_data[:half]
train_data = all_data
corpus_data = all_data[half:]

# --- Reformat corpus entries ---
def qa_to_doc(entry):
    return {
        "text": f"The answer to: {entry['prompt']} is: {entry['response']}",
        "source": entry["source"]
    }

corpus_docs = [qa_to_doc(e) for e in corpus_data]


def save_jsonl(path, data):
    with open(path, "w") as f:
        for row in data:
            json.dump(row, f)
            f.write("\n")

save_jsonl(output_dir / "train_qa.jsonl", [{"prompt": e["prompt"], "response": e["response"], "source": e["source"]} for e in train_data])
save_jsonl(output_dir / "corpus_docs.jsonl", corpus_docs)

print("Saved:")
print(f"- {output_dir/'train_qa.jsonl'}: {len(train_data)} entries")
print(f"- {output_dir/'corpus_docs.jsonl'}: {len(corpus_docs)} entries")


file_names = ['wiki_outputs/math_wiki.jsonl',
 'wiki_outputs/ds_wiki.jsonl',
 'wiki_outputs/cs_wiki.jsonl',
 'wiki_outputs/phy_wiki.jsonl',
 'wiki_outputs/eng_wiki.jsonl',
 'wiki_outputs/chem_wiki.jsonl',
 'wiki_outputs/biomed_wiki.jsonl',
 'shervine_cheatsheets_cleaned.jsonl',
 f'{output_dir}/corpus_docs.jsonl',
]

# Combine all domain-specific JSONL files
combined_data = []
for file in file_names:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            combined_data.append({
                "text": item["text"],
                "source": item["source"]
            })

# push corpus to HF Hub
dataset = Dataset.from_list(combined_data)
# dataset.push_to_hub("sucharush/stem_corpus")

# push half camel to HF Hub for sft
with open(f"{output_dir}/train_qa.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]
    
dataset = Dataset.from_list(train_data)
# dataset.push_to_hub("sucharush/MNLP_M3_rag_dataset", config_name="camel_subset")
