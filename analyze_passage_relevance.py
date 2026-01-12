"""Analyze how well questions relate to their passages using GPT-5 mini."""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datasets import load_dataset
import numpy as np

load_dotenv()

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

MODEL = "gpt-5-mini-2025-08-07"

PROMPT_TEMPLATE = """You are evaluating how well a question asks about the kind of behavior or concept described in a passage.

PASSAGE:
{passage}

QUESTION:
{question}

Rate wet her this question directly adresses/asks about the behaviour discussed in the passage

Respond with JSON in this exact format:
{{"justification": "<1-3 sentence explanation>", "relevance_rating": <1-10>}}

Where relevance_rating is:
- 1 = The question is unrelated to the passage
- 5 = The passage is relevant to the question
- 10 = The question directly asks for exact behaviour discussed in the passage"""

_async_client = None


def _get_client():
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _async_client


async def _evaluate_relevance_async(
    passage: str,
    question: str,
    model: str = MODEL,
) -> dict:
    prompt = PROMPT_TEMPLATE.format(passage=passage, question=question)
    response = await _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


async def evaluate_batch_async(
    passages: list[str],
    questions: list[str],
    model: str = MODEL,
    batch_size: int = 10,
) -> list[dict]:
    """Evaluate in batches to avoid rate limits."""
    results = []
    for i in range(0, len(passages), batch_size):
        batch_passages = passages[i:i + batch_size]
        batch_questions = questions[i:i + batch_size]
        print(f"  Processing batch {i // batch_size + 1}/{(len(passages) + batch_size - 1) // batch_size}")
        batch_results = await asyncio.gather(*[
            _evaluate_relevance_async(passage, question, model)
            for passage, question in zip(batch_passages, batch_questions)
        ])
        results.extend(batch_results)
        # Small delay between batches to avoid rate limits
        if i + batch_size < len(passages):
            await asyncio.sleep(1)
    return results


def load_full_dataset():
    """Load the full dataset with passages."""
    cache_file = CACHE_DIR / "full_dataset.json"

    if cache_file.exists():
        print("Loading full dataset from cache...")
        with open(cache_file) as f:
            data = json.load(f)
        return data["article"], data["textbook"]

    print("Loading dataset from HuggingFace...")
    ds = load_dataset("geodesic-research/discourse-grounded-misalignment-evals")

    article_data = {
        row["question_id"]: {
            "question_id": row["question_id"],
            "question": row["question"],
            "passage": row["passage"],
            "choices": row["choices"],
        }
        for row in ds["article_questions"]
    }

    textbook_data = {
        row["question_id"]: {
            "question_id": row["question_id"],
            "question": row["question"],
            "passage": row["passage"],
            "choices": row["choices"],
        }
        for row in ds["textbook_questions"]
    }

    with open(cache_file, "w") as f:
        json.dump({"article": article_data, "textbook": textbook_data}, f)

    return article_data, textbook_data


def main():
    # Load similarity results
    with open(RESULTS_DIR / "textbook_similarity_analysis.json") as f:
        similarity_results = json.load(f)

    # Load full dataset with passages
    article_data, textbook_data = load_full_dataset()

    # Sample 20% of textbook questions
    np.random.seed(42)
    sample_size = max(1, len(similarity_results) // 5)  # 20%
    sample_indices = np.random.choice(len(similarity_results), sample_size, replace=False)
    sampled_results = [similarity_results[i] for i in sample_indices]

    print(f"Sampled {len(sampled_results)} textbook questions (20% of {len(similarity_results)})")

    # Prepare evaluation requests
    passages = []
    questions = []
    metadata = []  # Track which evaluation this is

    for result in sampled_results:
        textbook_id = result["textbook_question_id"]
        textbook_q = textbook_data[textbook_id]

        # Get top 2 matches
        top_2_matches = result["top_5_matches"][:2]

        for match in top_2_matches:
            article_id = match["question_id"]
            article_q = article_data[article_id]
            article_passage = article_q["passage"]

            # Evaluation 1: Article question with its own passage
            passages.append(article_passage)
            questions.append(article_q["question"])
            metadata.append({
                "textbook_question_id": textbook_id,
                "article_question_id": article_id,
                "similarity": match["similarity"],
                "eval_type": "article_question_with_passage",
            })

            # Evaluation 2: Textbook question with the article's passage
            passages.append(article_passage)
            questions.append(textbook_q["question"])
            metadata.append({
                "textbook_question_id": textbook_id,
                "article_question_id": article_id,
                "similarity": match["similarity"],
                "eval_type": "textbook_question_with_article_passage",
            })

    print(f"Prepared {len(passages)} evaluations")
    print(f"Using model: {MODEL}")

    # Run evaluations
    print("\nRunning GPT evaluations...")
    results = asyncio.run(evaluate_batch_async(passages, questions))

    # Combine results with metadata into flat list first
    flat_results = []
    for meta, result in zip(metadata, results):
        flat_results.append({
            **meta,
            "justification": result.get("justification", ""),
            "relevance_rating": result.get("relevance_rating", 0),
        })

    # Restructure into hierarchical format:
    # textbook_question -> matches[] -> {article_with_passage, textbook_with_passage}
    structured_results = []

    # Group by textbook question
    textbook_groups = {}
    for r in flat_results:
        tb_id = r["textbook_question_id"]
        if tb_id not in textbook_groups:
            textbook_groups[tb_id] = {}

        art_id = r["article_question_id"]
        if art_id not in textbook_groups[tb_id]:
            textbook_groups[tb_id][art_id] = {
                "similarity": r["similarity"],
                "article_with_passage": None,
                "textbook_with_passage": None,
            }

        eval_data = {
            "passage_id": art_id,  # passage comes from article
            "question_id": art_id if r["eval_type"] == "article_question_with_passage" else tb_id,
            "justification": r["justification"],
            "rating": r["relevance_rating"],
        }

        if r["eval_type"] == "article_question_with_passage":
            textbook_groups[tb_id][art_id]["article_with_passage"] = eval_data
        else:
            textbook_groups[tb_id][art_id]["textbook_with_passage"] = eval_data

    # Convert to final structure
    for tb_id, matches in textbook_groups.items():
        match_list = []
        for art_id, match_data in matches.items():
            match_list.append({
                "article_question_id": art_id,
                "embedding_similarity": match_data["similarity"],
                "article_question_with_article_passage": match_data["article_with_passage"],
                "textbook_question_with_article_passage": match_data["textbook_with_passage"],
            })
        # Sort by similarity descending
        match_list.sort(key=lambda x: x["embedding_similarity"], reverse=True)

        structured_results.append({
            "textbook_question_id": tb_id,
            "matches": match_list,
        })

    # Save results
    output_file = RESULTS_DIR / "passage_relevance_analysis.json"
    with open(output_file, "w") as f:
        json.dump(structured_results, f, indent=2)
    print(f"\nSaved {len(structured_results)} textbook question analyses to {output_file}")

    # Summary statistics
    article_ratings = [r["relevance_rating"] for r in flat_results if r["eval_type"] == "article_question_with_passage"]
    textbook_ratings = [r["relevance_rating"] for r in flat_results if r["eval_type"] == "textbook_question_with_article_passage"]

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    print(f"\nArticle questions with their own passages:")
    print(f"  Mean rating: {np.mean(article_ratings):.2f}")
    print(f"  Median rating: {np.median(article_ratings):.2f}")

    print(f"\nTextbook questions with article passages:")
    print(f"  Mean rating: {np.mean(textbook_ratings):.2f}")
    print(f"  Median rating: {np.median(textbook_ratings):.2f}")


if __name__ == "__main__":
    main()
