# Alignment Pretraining Data Investigation

This project investigates the validity of using textbook questions as an evaluation dataset when training on article data from the [discourse-grounded-misalignment-evals](https://huggingface.co/datasets/geodesic-research/discourse-grounded-misalignment-evals) dataset.

## The Problem

When training a model on one data source (articles) and evaluating on questions from another source (textbooks), we need to ensure the evaluation is meaningful. Specifically:

**If a textbook question could just as easily be answered using information from the article training data, it doesn't actually test whether the model learned from the textbook content.**

For example, if a textbook question asks about "deceptive alignment" and the article training data also contains passages about deceptive alignment, a model could answer correctly by learning from the articles alone - making the textbook question useless as an out-of-distribution evaluation.

## Methodology

### Step 1: Find Similar Question Pairs

We use sentence embeddings (`all-MiniLM-L6-v2`) to find the most semantically similar article questions for each textbook question. This identifies cases where the topics overlap.

```
investigate.py → results/textbook_similarity_analysis.json
```

### Step 2: Evaluate Passage-Question Relevance

For each textbook question, we take its top 2 most similar article questions and evaluate:

1. **Article question with its own passage**: Does the article question actually relate to its passage? (sanity check - should be high)

2. **Textbook question with the article passage**: Could the textbook question be answered using the article's passage?

We use GPT-5-mini to rate relevance on a 1-10 scale:
- 1 = The question is unrelated to the passage
- 5 = The passage is relevant to the question
- 10 = The question directly asks about behavior discussed in the passage

```
analyze_passage_relevance.py → results/passage_relevance_analysis.json
```

### Step 3: Generate Summary

We extract the relevance rating for each textbook question, sorted by rating.

```
create_summary.py → results/textbook_relevance_summary.csv
```

## Results

The summary CSV (`results/textbook_relevance_summary.csv`) shows each textbook question ID and its relevance rating.

- **High ratings (8-10)**: The textbook question could potentially be answered using article training data - problematic for evaluation validity
- **Low ratings (1-4)**: The textbook question covers content not found in similar article passages - good for out-of-distribution evaluation

## Files

| File | Description |
|------|-------------|
| `investigate.py` | Computes embedding similarities between textbook and article questions |
| `analyze_passage_relevance.py` | Uses GPT to evaluate passage-question relevance |
| `create_summary.py` | Generates the summary CSV |
| `plot_relevance.py` | Creates visualization plots |
| `results/textbook_relevance_summary.csv` | **Main output**: textbook question IDs with relevance ratings |
| `results/passage_relevance_analysis.json` | Full analysis with justifications |
| `results/plots/` | Distribution and example comparison plots |

## Usage

```bash
# Install dependencies
uv sync

# Run the full pipeline
uv run python investigate.py              # Step 1: Find similar pairs
uv run python analyze_passage_relevance.py  # Step 2: GPT relevance analysis
uv run python create_summary.py           # Step 3: Generate summary CSV
uv run python plot_relevance.py           # Optional: Generate plots
```

## Requirements

- Python 3.11+
- OpenAI API key (set in `.env` as `OPENAI_API_KEY`)
