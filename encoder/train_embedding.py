from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.datasets import NoDuplicatesDataLoader
from datasets import load_dataset
import os

# -------- Convert to InputExample --------
def to_input_examples(split):
    return [
        InputExample(
            texts=[
                "Represent this question for retrieving relevant documents: " + ex["query"],
                ex["positive"]
            ]
        )
        for ex in split
    ]

# -------- Custom Evaluator --------
class SimpleEvaluator(evaluation.InformationRetrievalEvaluator):
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        return super().__call__(model, output_path, epoch, steps)

def main():
    # -------- Config --------
    model_name = "thenlper/gte-small"
    dataset_name = ("sucharush/MNLP_M3_rag_dataset", "embedding_data")
    output_dir = "./gte_stem_finetuned"
    eval_split = "test"
    batch_size = 32
    epochs = 1

    # -------- Load model --------
    model = SentenceTransformer(model_name)
    model._first_module().pooling_mode_mean_tokens = True  # force mean pooling

    # -------- Load dataset --------
    dataset = load_dataset(*dataset_name)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]

    train_examples = to_input_examples(train_data)
    train_dataloader = NoDuplicatesDataLoader(train_examples, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # -------- Precision@k Evaluator --------
    eval_corpus = {str(i): ex["positive"] for i, ex in enumerate(eval_data)}
    eval_queries = {str(i): ex["query"] for i, ex in enumerate(eval_data)}
    relevant_docs = {str(i): set([str(i)]) for i in range(len(eval_data))}

    retrieval_evaluator = SimpleEvaluator(
        queries=eval_queries,
        corpus=eval_corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=True,
        precision_recall_at_k=[1, 3, 5],
        name="ir-eval"
    )

    # -------- Training --------
    print("Starting to train the encoder...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=retrieval_evaluator,
        epochs=epochs,
        evaluation_steps=500,
        warmup_steps=100,
        output_path=output_dir,
        show_progress_bar=True,
        checkpoint_path=output_dir + "/checkpoints",
    )

    # -------- Save final model --------
    model.half()
    # model.push_to_hub(repo_id="sucharush/MNLP_M3_document_encoder", exist_ok=True)

if __name__ == "__main__":
    main()
