"""Investigate how OOD textbook questions are relative to article questions."""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Avoid xet download issues

import json
from pathlib import Path
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"


def load_questions_with_topics():
    """Load and cache the questions, choices, and topics from both splits."""
    cache_file = CACHE_DIR / "questions_with_topics_and_choices.json"

    if cache_file.exists():
        print("Loading questions with topics from cache...")
        with open(cache_file) as f:
            data = json.load(f)
        return data["article"], data["textbook"]

    print("Loading dataset from HuggingFace...")
    ds = load_dataset("geodesic-research/discourse-grounded-misalignment-evals")

    article_data = [
        {"question_id": qid, "question": q, "choices": c, "topics": t}
        for qid, q, c, t in zip(
            ds["article_questions"]["question_id"],
            ds["article_questions"]["question"],
            ds["article_questions"]["choices"],
            ds["article_questions"]["topics"],
        )
    ]
    textbook_data = [
        {"question_id": qid, "question": q, "choices": c, "topics": t}
        for qid, q, c, t in zip(
            ds["textbook_questions"]["question_id"],
            ds["textbook_questions"]["question"],
            ds["textbook_questions"]["choices"],
            ds["textbook_questions"]["topics"],
        )
    ]

    with open(cache_file, "w") as f:
        json.dump({"article": article_data, "textbook": textbook_data}, f)

    return article_data, textbook_data


def get_embeddings(questions: list[str], name: str, model: SentenceTransformer) -> np.ndarray:
    """Get embeddings, loading from cache if available."""
    cache_file = CACHE_DIR / f"{name}_embeddings.npy"

    if cache_file.exists():
        print(f"Loading {name} embeddings from cache...")
        return np.load(cache_file)

    print(f"Computing {name} embeddings ({len(questions)} questions)...")
    embeddings = model.encode(questions, show_progress_bar=True, device="mps")

    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    np.save(cache_file, embeddings)
    return embeddings


def find_nearest_neighbors(query_emb: np.ndarray, corpus_embs: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors by cosine similarity."""
    similarities = corpus_embs @ query_emb
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return top_k_indices, similarities[top_k_indices]


def find_top_k_matches(
    article_data: list[dict],
    article_embs: np.ndarray,
    textbook_emb: np.ndarray,
    k: int = 5,
) -> list[dict]:
    """Find top k matching article questions by cosine similarity."""
    all_similarities = article_embs @ textbook_emb
    top_k_indices = np.argsort(all_similarities)[-k:][::-1]

    return [
        {
            "question_id": article_data[idx]["question_id"],
            "similarity": float(all_similarities[idx]),
        }
        for idx in top_k_indices
    ]


def main():
    article_data, textbook_data = load_questions_with_topics()
    article_qs = [item["question"] for item in article_data]
    textbook_qs = [item["question"] for item in textbook_data]

    print(f"Article questions: {len(article_data)}")
    print(f"Textbook questions: {len(textbook_data)}")

    print(f"\nLoading model ({MODEL_NAME}) on MPS...")
    model = SentenceTransformer(MODEL_NAME, device="mps")

    article_embs = get_embeddings(article_qs, "article", model)
    textbook_embs = get_embeddings(textbook_qs, "textbook", model)

    # Process ALL textbook questions
    print(f"\nProcessing {len(textbook_data)} textbook questions...")

    results = []
    best_sims = []

    for idx, textbook_item in enumerate(textbook_data):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(textbook_data)}")

        top_matches = find_top_k_matches(
            article_data,
            article_embs,
            textbook_embs[idx],
            k=5,
        )

        best_sims.append(top_matches[0]["similarity"])

        results.append({
            "textbook_question_id": textbook_item["question_id"],
            "top_5_matches": top_matches,
        })

    # Save results
    output_file = RESULTS_DIR / "textbook_similarity_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_file}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    best_sims = np.array(best_sims)

    print(f"\nBEST MATCH SIMILARITIES:")
    print(f"  Mean:   {np.mean(best_sims):.3f}")
    print(f"  Median: {np.median(best_sims):.3f}")
    print(f"  Std:    {np.std(best_sims):.3f}")
    print(f"  Min:    {np.min(best_sims):.3f}")
    print(f"  Max:    {np.max(best_sims):.3f}")
    print(f"  % < 0.5: {100 * np.mean(best_sims < 0.5):.1f}%")
    print(f"  % < 0.7: {100 * np.mean(best_sims < 0.7):.1f}%")


if __name__ == "__main__":
    main()
