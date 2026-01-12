"""Create a simple summary CSV for paper authors to filter textbook questions by relevance."""

import json
import csv
from pathlib import Path

RESULTS_DIR = Path("results")


def main():
    with open(RESULTS_DIR / "passage_relevance_analysis.json") as f:
        relevance_results = json.load(f)

    rows = []
    for item in relevance_results:
        textbook_id = item["textbook_question_id"]

        # Get the best match (highest article rating with its own passage)
        best_match = None
        best_article_rating = -1

        for match in item["matches"]:
            article_eval = match["article_question_with_article_passage"]
            if article_eval and article_eval["rating"] > best_article_rating:
                best_article_rating = article_eval["rating"]
                best_match = match

        if best_match:
            textbook_eval = best_match["textbook_question_with_article_passage"]
            textbook_rating = textbook_eval["rating"] if textbook_eval else 0
            textbook_justification = textbook_eval["justification"] if textbook_eval else ""
        else:
            textbook_rating = 0
            textbook_justification = ""

        rows.append({
            "textbook_question_id": textbook_id,
            "relevance_rating": textbook_rating,
        })

    # Sort by relevance rating descending
    rows.sort(key=lambda x: x["relevance_rating"], reverse=True)

    # Write CSV
    output_path = RESULTS_DIR / "textbook_relevance_summary.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["textbook_question_id", "relevance_rating"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary to {output_path}")
    print(f"Total questions: {len(rows)}")
    print(f"Rating distribution:")
    for rating in range(10, 0, -1):
        count = sum(1 for r in rows if r["relevance_rating"] == rating)
        if count > 0:
            print(f"  {rating}: {count}")


if __name__ == "__main__":
    main()
