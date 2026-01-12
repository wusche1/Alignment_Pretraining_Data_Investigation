"""Analyze how well questions relate to their passages using GPT-5 mini."""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import json
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datasets import load_dataset

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
) -> list[dict]:
    """Evaluate all passages/questions in parallel."""
    return await asyncio.gather(*[
        _evaluate_relevance_async(passage, question, model)
        for passage, question in zip(passages, questions)
    ])


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


def load_existing_results():
    """Load existing results and return set of already processed textbook IDs."""
    output_file = RESULTS_DIR / "passage_relevance_analysis.json"
    if output_file.exists():
        with open(output_file) as f:
            results = json.load(f)
        return {r["textbook_question_id"] for r in results}
    return set()


def append_result(result: dict):
    """Append a single result to the JSON file."""
    output_file = RESULTS_DIR / "passage_relevance_analysis.json"

    if output_file.exists():
        with open(output_file) as f:
            results = json.load(f)
    else:
        results = []

    results.append(result)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


async def process_single_textbook_question(
    textbook_id: str,
    textbook_q: dict,
    top_matches: list,
    article_data: dict,
) -> dict:
    """Process a single textbook question and return structured result."""
    passages = []
    questions = []
    metadata = []

    for match in top_matches:
        article_id = match["question_id"]
        article_q = article_data[article_id]
        article_passage = article_q["passage"]

        # Evaluation 1: Article question with its own passage
        passages.append(article_passage)
        questions.append(article_q["question"])
        metadata.append({
            "article_question_id": article_id,
            "similarity": match["similarity"],
            "eval_type": "article_question_with_passage",
        })

        # Evaluation 2: Textbook question with the article's passage
        passages.append(article_passage)
        questions.append(textbook_q["question"])
        metadata.append({
            "article_question_id": article_id,
            "similarity": match["similarity"],
            "eval_type": "textbook_question_with_article_passage",
        })

    # Run evaluations for this textbook question
    results = await evaluate_batch_async(passages, questions)

    # Structure the results
    matches_dict = {}
    for meta, result in zip(metadata, results):
        art_id = meta["article_question_id"]
        if art_id not in matches_dict:
            matches_dict[art_id] = {
                "similarity": meta["similarity"],
                "article_with_passage": None,
                "textbook_with_passage": None,
            }

        eval_data = {
            "passage_id": art_id,
            "question_id": art_id if meta["eval_type"] == "article_question_with_passage" else textbook_id,
            "justification": result.get("justification", ""),
            "rating": result.get("relevance_rating", 0),
        }

        if meta["eval_type"] == "article_question_with_passage":
            matches_dict[art_id]["article_with_passage"] = eval_data
        else:
            matches_dict[art_id]["textbook_with_passage"] = eval_data

    # Convert to final structure
    match_list = []
    for art_id, match_data in matches_dict.items():
        match_list.append({
            "article_question_id": art_id,
            "embedding_similarity": match_data["similarity"],
            "article_question_with_article_passage": match_data["article_with_passage"],
            "textbook_question_with_article_passage": match_data["textbook_with_passage"],
        })
    match_list.sort(key=lambda x: x["embedding_similarity"], reverse=True)

    return {
        "textbook_question_id": textbook_id,
        "matches": match_list,
    }


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


async def process_batch_of_questions(
    batch: list[dict],
    textbook_data: dict,
    article_data: dict,
) -> list[dict]:
    """Process a batch of textbook questions in parallel."""
    tasks = []
    for result in batch:
        textbook_id = result["textbook_question_id"]
        textbook_q = textbook_data[textbook_id]
        top_2_matches = result["top_5_matches"][:2]
        tasks.append(process_single_textbook_question(
            textbook_id, textbook_q, top_2_matches, article_data
        ))
    return await asyncio.gather(*tasks)


async def main_async():
    # Load similarity results
    with open(RESULTS_DIR / "textbook_similarity_analysis.json") as f:
        similarity_results = json.load(f)

    # Load full dataset with passages
    article_data, textbook_data = load_full_dataset()

    # Load already processed IDs
    processed_ids = load_existing_results()
    total_questions = len(similarity_results)
    skipped = len(processed_ids)

    # Filter to unprocessed
    to_process = [r for r in similarity_results if r["textbook_question_id"] not in processed_ids]
    remaining = len(to_process)

    print(f"\n{'='*60}")
    print(f"Total textbook questions: {total_questions}")
    print(f"Already processed (skipping): {skipped}")
    print(f"Remaining to process: {remaining}")
    print(f"Model: {MODEL}")
    print(f"{'='*60}\n")

    if remaining == 0:
        print("Nothing to process!")
        return

    batch_size = 50
    processed_count = 0
    start_time = time.time()

    for batch_start in range(0, remaining, batch_size):
        batch_end = min(batch_start + batch_size, remaining)
        batch = to_process[batch_start:batch_end]

        batch_start_time = time.time()
        results = await process_batch_of_questions(batch, textbook_data, article_data)

        for result in results:
            append_result(result)

        processed_count += len(batch)
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - start_time

        # Calculate ETA
        avg_time_per_item = elapsed / processed_count
        remaining_items = remaining - processed_count
        eta_seconds = avg_time_per_item * remaining_items

        print(f"[{skipped + processed_count}/{total_questions}] "
              f"Batch of {len(batch)} done in {batch_time:.1f}s | "
              f"Elapsed: {format_time(elapsed)} | "
              f"ETA: {format_time(eta_seconds)} | "
              f"Speed: {processed_count / elapsed:.1f}/s")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done! Processed {processed_count} questions in {format_time(total_time)}")
    print(f"Total in database: {skipped + processed_count}")
    print(f"{'='*60}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
