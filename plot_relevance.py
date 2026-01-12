"""Plot relevance rating distributions and show example comparisons."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
CACHE_DIR = Path("cache")

# Colors
COLOR_TEXTBOOK = "#E74C3C"  # Red
COLOR_ARTICLE = "#3498DB"   # Blue
COLOR_PASSAGE = "#F5F5DC"   # Beige
COLOR_QUESTION = "#E8F4FD"  # Light blue

# Number of random samples to show
NUM_RANDOM_SAMPLES = 5


def load_data():
    """Load the relevance analysis results and full dataset."""
    with open(RESULTS_DIR / "passage_relevance_analysis.json") as f:
        relevance_results = json.load(f)

    with open(CACHE_DIR / "full_dataset.json") as f:
        full_data = json.load(f)

    return relevance_results, full_data["article"], full_data["textbook"]


def get_best_match_per_textbook(relevance_results):
    """
    For each textbook question, select the match where the article passage
    has the highest rating with its own question.
    """
    best_matches = []

    for item in relevance_results:
        textbook_id = item["textbook_question_id"]
        matches = item["matches"]

        best_match = None
        best_article_rating = -1

        for match in matches:
            article_eval = match["article_question_with_article_passage"]
            if article_eval and article_eval["rating"] > best_article_rating:
                best_article_rating = article_eval["rating"]
                best_match = match

        if best_match:
            textbook_eval = best_match["textbook_question_with_article_passage"]
            best_matches.append({
                "textbook_question_id": textbook_id,
                "article_question_id": best_match["article_question_id"],
                "embedding_similarity": best_match["embedding_similarity"],
                "article_rating": best_match["article_question_with_article_passage"]["rating"],
                "textbook_rating": textbook_eval["rating"] if textbook_eval else 0,
                "article_justification": best_match["article_question_with_article_passage"]["justification"],
                "textbook_justification": textbook_eval["justification"] if textbook_eval else "",
            })

    return best_matches


def get_random_samples(best_matches, n=None):
    """Get n random samples from best matches."""
    if n is None:
        n = NUM_RANDOM_SAMPLES
    import random
    indices = random.sample(range(len(best_matches)), min(n, len(best_matches)))
    return [best_matches[i] for i in indices]


def wrap_text(text, width=80):
    """Wrap text to specified width."""
    return "\n".join(wrap(text, width))


def plot_distribution(best_matches):
    """Create the distribution plot with side-by-side bars."""
    PLOTS_DIR.mkdir(exist_ok=True)

    article_ratings = [m["article_rating"] for m in best_matches]
    textbook_ratings = [m["textbook_rating"] for m in best_matches]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Count occurrences for each rating
    ratings = np.arange(1, 11)
    article_counts = [article_ratings.count(r) for r in ratings]
    textbook_counts = [textbook_ratings.count(r) for r in ratings]

    # Bar width and positions
    bar_width = 0.35
    x = np.arange(len(ratings))

    bars1 = ax.bar(x - bar_width/2, article_counts, bar_width,
                   label='Article Q with Article Passage', color=COLOR_ARTICLE, edgecolor='black')
    bars2 = ax.bar(x + bar_width/2, textbook_counts, bar_width,
                   label='Textbook Q with Article Passage', color=COLOR_TEXTBOOK, edgecolor='black')

    ax.set_xlabel('Relevance Rating (1-10)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Passage-Question Relevance Ratings\n(Best match per textbook question by article rating)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ratings)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add statistics text
    stats_text = (
        f"Article Q: Mean={np.mean(article_ratings):.2f}, Median={np.median(article_ratings):.1f}\n"
        f"Textbook Q: Mean={np.mean(textbook_ratings):.2f}, Median={np.median(textbook_ratings):.1f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "relevance_distribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved distribution plot to {PLOTS_DIR / 'relevance_distribution.png'}")
    plt.close()


def plot_all_examples(samples, article_data, textbook_data):
    """Create a single clean figure with all examples - structured layout."""
    # Build content for each example
    content_blocks = []
    for idx, example in enumerate(samples):
        art_id = example["article_question_id"]
        tb_id = example["textbook_question_id"]
        article = article_data[art_id]
        textbook = textbook_data[tb_id]

        # Format choices
        tb_choices = "\n".join(f"  {i+1}. {wrap_text(c, 55)}" for i, c in enumerate(textbook['choices']))
        art_choices = "\n".join(f"  {i+1}. {wrap_text(c, 55)}" for i, c in enumerate(article['choices']))

        content_blocks.append({
            'sample_num': idx + 1,
            'similarity': example['embedding_similarity'],
            'article_rating': example['article_rating'],
            'textbook_rating': example['textbook_rating'],
            'textbook_passage': wrap_text(textbook['passage'], 60),
            'textbook_question': wrap_text(textbook['question'], 55),
            'textbook_choices': tb_choices,
            'article_passage': wrap_text(article['passage'], 60),
            'article_question': wrap_text(article['question'], 55),
            'article_choices': art_choices,
        })

    # Dynamic sizing based on number of examples
    n_examples = len(samples)
    n_rows = n_examples * 3  # header + passage + qa per example
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 6 * n_examples),
                              gridspec_kw={'height_ratios': [0.15, 0.4, 0.45] * n_examples,
                                          'hspace': 0.01, 'wspace': 0.02})

    # Turn off all axes initially
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    for idx, block in enumerate(content_blocks):
        base_row = idx * 3

        # Row 0: Header - merge both columns by hiding right one
        ax_header = axes[base_row, 0]
        axes[base_row, 1].set_visible(False)
        header_text = (
            f"SAMPLE {block['sample_num']}  |  "
            f"Similarity: {block['similarity']:.3f}  |  "
            f"Article Rating: {block['article_rating']}/10  |  "
            f"Textbook Rating: {block['textbook_rating']}/10"
        )
        ax_header.text(0.5, 0.5, header_text, transform=ax_header.transAxes,
                       fontsize=10, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                                edgecolor='#F9A825', linewidth=2))

        # Row 1: Passages
        axes[base_row + 1, 0].text(0.02, 0.95, f"TEXTBOOK PASSAGE:\n\n{block['textbook_passage']}",
                                    transform=axes[base_row + 1, 0].transAxes, fontsize=8,
                                    verticalalignment='top', fontfamily='monospace',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                                             edgecolor=COLOR_TEXTBOOK, linewidth=1.5))

        axes[base_row + 1, 1].text(0.02, 0.95, f"ARTICLE PASSAGE:\n\n{block['article_passage']}",
                                    transform=axes[base_row + 1, 1].transAxes, fontsize=8,
                                    verticalalignment='top', fontfamily='monospace',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                                             edgecolor=COLOR_ARTICLE, linewidth=1.5))

        # Row 2: Questions + Choices
        tb_qa_text = f"TEXTBOOK QUESTION:\n{block['textbook_question']}\n\nCHOICES:\n{block['textbook_choices']}"
        axes[base_row + 2, 0].text(0.02, 0.95, tb_qa_text,
                                    transform=axes[base_row + 2, 0].transAxes, fontsize=8,
                                    verticalalignment='top', fontfamily='monospace',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFCDD2',
                                             edgecolor=COLOR_TEXTBOOK, linewidth=1.5))

        art_qa_text = f"ARTICLE QUESTION:\n{block['article_question']}\n\nCHOICES:\n{block['article_choices']}"
        axes[base_row + 2, 1].text(0.02, 0.95, art_qa_text,
                                    transform=axes[base_row + 2, 1].transAxes, fontsize=8,
                                    verticalalignment='top', fontfamily='monospace',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#BBDEFB',
                                             edgecolor=COLOR_ARTICLE, linewidth=1.5))

    plt.savefig(PLOTS_DIR / "examples_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved examples plot to {PLOTS_DIR / 'examples_comparison.png'}")
    plt.close()


def create_example_text(example, article_data, textbook_data):
    """Create formatted text for an example comparison."""
    art_id = example["article_question_id"]
    tb_id = example["textbook_question_id"]

    article = article_data[art_id]
    textbook = textbook_data[tb_id]

    text = f"""
{'='*100}
TEXTBOOK QUESTION ID: {tb_id}
ARTICLE QUESTION ID: {art_id}
Embedding Similarity: {example['embedding_similarity']:.3f}
Article Rating: {example['article_rating']}/10 | Textbook Rating: {example['textbook_rating']}/10
{'='*100}

ARTICLE PASSAGE:
{'-'*50}
{wrap_text(article['passage'])}

ARTICLE QUESTION:
{'-'*50}
{wrap_text(article['question'])}

ARTICLE CHOICES:
"""
    for i, choice in enumerate(article['choices']):
        text += f"  {i+1}. {wrap_text(choice, 90)}\n"

    text += f"""
{'='*100}

TEXTBOOK PASSAGE:
{'-'*50}
{wrap_text(textbook['passage'])}

TEXTBOOK QUESTION:
{'-'*50}
{wrap_text(textbook['question'])}

TEXTBOOK CHOICES:
"""
    for i, choice in enumerate(textbook['choices']):
        text += f"  {i+1}. {wrap_text(choice, 90)}\n"

    text += f"""
{'='*100}
GPT JUSTIFICATIONS:
Article question relevance: {example['article_justification']}
Textbook question relevance: {example['textbook_justification']}
{'='*100}
"""
    return text


def main():
    print("Loading data...")
    relevance_results, article_data, textbook_data = load_data()

    print("Processing best matches...")
    best_matches = get_best_match_per_textbook(relevance_results)

    article_ratings = [m["article_rating"] for m in best_matches]
    textbook_ratings = [m["textbook_rating"] for m in best_matches]

    print(f"Total matches: {len(best_matches)}")
    print(f"Article ratings - Mean: {np.mean(article_ratings):.2f}, Median: {np.median(article_ratings):.2f}")
    print(f"Textbook ratings - Mean: {np.mean(textbook_ratings):.2f}, Median: {np.median(textbook_ratings):.2f}")

    # Create plots directory
    PLOTS_DIR.mkdir(exist_ok=True)

    # Plot 1: Distribution
    print("\nCreating distribution plot...")
    plot_distribution(best_matches)

    # Plot 2: All examples in one file
    print("\nCreating examples plot...")
    samples = get_random_samples(best_matches)
    plot_all_examples(samples, article_data, textbook_data)

    # Save detailed examples to text file
    print("\nSaving detailed examples to text file...")
    with open(RESULTS_DIR / "example_comparisons.txt", "w") as f:
        f.write("PASSAGE-QUESTION RELEVANCE EXAMPLES (RANDOM SAMPLES)\n")
        f.write("=" * 100 + "\n\n")

        for idx, sample in enumerate(samples):
            f.write(f"\n\n{'#'*100}\n")
            f.write(f"# SAMPLE {idx + 1}\n")
            f.write(f"{'#'*100}\n")
            f.write(create_example_text(sample, article_data, textbook_data))

    print(f"Saved detailed examples to {RESULTS_DIR / 'example_comparisons.txt'}")
    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
