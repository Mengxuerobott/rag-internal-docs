"""
eval/ragas_eval.py
──────────────────
Evaluates the RAG pipeline using the RAGAS framework.

RAGAS metrics explained (use these in interviews):
───────────────────────────────────────────────────
  faithfulness        — Does the answer contain only claims supported by the
                        retrieved context? (0=hallucinating, 1=fully grounded)

  answer_relevancy    — Is the generated answer actually relevant to the question?
                        (penalises rambling / off-topic answers)

  context_precision   — Are the retrieved chunks actually useful for answering
                        the question? (penalises noisy retrieval)

  context_recall      — Did the retrieval surface all the information needed
                        to answer the question? (requires ground-truth answers)

Target scores for a strong portfolio:
  faithfulness ≥ 0.88, answer_relevancy ≥ 0.85, context_precision ≥ 0.80

Run:
    python -m eval.ragas_eval
    python -m eval.ragas_eval --output results/ragas_scores.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from datasets import Dataset
from loguru import logger
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from rich.console import Console
from rich.table import Table


# ── Test dataset ──────────────────────────────────────────────────────────────
# In production you'd load this from a JSON / CSV file.
# Ground-truth answers are written by humans who have read the actual documents.
# For your portfolio: write 10-20 question/answer pairs covering your sample docs.

DEFAULT_TEST_CASES = [
    {
        "question": "What is the company's parental leave policy?",
        "ground_truth": (
            "Employees are entitled to 16 weeks of fully paid parental leave "
            "for primary caregivers and 4 weeks for secondary caregivers, "
            "starting from the date of birth or adoption."
        ),
    },
    {
        "question": "How do I submit an expense report?",
        "ground_truth": (
            "Expense reports must be submitted through the Concur platform within "
            "30 days of the expense. Receipts over $25 are required. "
            "Manager approval is needed for amounts over $500."
        ),
    },
    {
        "question": "What is the password rotation policy?",
        "ground_truth": (
            "All employee passwords must be rotated every 90 days. "
            "Passwords must be at least 12 characters long and include "
            "uppercase, lowercase, a number, and a special character."
        ),
    },
    {
        "question": "What is the process for requesting time off?",
        "ground_truth": (
            "Time-off requests must be submitted at least 2 weeks in advance "
            "through the HR portal. Manager approval is required. "
            "Unused vacation days can be carried over up to 10 days."
        ),
    },
    {
        "question": "How long is the probationary period for new employees?",
        "ground_truth": (
            "New employees have a 90-day probationary period during which "
            "performance is reviewed monthly. Full benefits begin on day one."
        ),
    },
]


# ── Core evaluation function ──────────────────────────────────────────────────
def run_evaluation(
    test_cases: list[dict] | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Run the full RAGAS evaluation suite against the live RAG pipeline.

    Args:
        test_cases:   List of {"question": str, "ground_truth": str} dicts.
                      Defaults to DEFAULT_TEST_CASES.
        output_path:  Optional path to write JSON results file.

    Returns:
        Dictionary with metric names and scores.
    """
    from ingestion.embedder import get_or_build_index
    from retrieval.query_engine import get_query_engine

    cases = test_cases or DEFAULT_TEST_CASES
    logger.info(f"Running RAGAS evaluation on {len(cases)} test cases…")

    # Load the query engine
    index = get_or_build_index()
    engine = get_query_engine(index)

    # ── Collect pipeline responses ────────────────────────────────────────────
    eval_rows = []
    for i, case in enumerate(cases, 1):
        question = case["question"]
        ground_truth = case.get("ground_truth", "")

        logger.info(f"  [{i}/{len(cases)}] {question!r}")

        start = time.perf_counter()
        try:
            response = engine.query(question)
            answer = str(response)
            contexts = [node.node.text for node in response.source_nodes]
            latency = time.perf_counter() - start

            eval_rows.append({
                "question":     question,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": ground_truth,
                "latency_s":    round(latency, 2),
            })

            logger.debug(f"    → answered in {latency:.1f}s, {len(contexts)} context chunks")

        except Exception as e:
            logger.error(f"    → FAILED: {e}")
            eval_rows.append({
                "question":     question,
                "answer":       f"ERROR: {e}",
                "contexts":     [],
                "ground_truth": ground_truth,
                "latency_s":    0.0,
            })

    # ── Build RAGAS dataset ───────────────────────────────────────────────────
    dataset = Dataset.from_list([
        {
            "question":     row["question"],
            "answer":       row["answer"],
            "contexts":     row["contexts"],
            "ground_truth": row["ground_truth"],
        }
        for row in eval_rows
    ])

    # ── Run RAGAS metrics ─────────────────────────────────────────────────────
    logger.info("Computing RAGAS metrics…")
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        raise_exceptions=False,
    )

    results = {
        "faithfulness":      round(float(scores["faithfulness"]), 4),
        "answer_relevancy":  round(float(scores["answer_relevancy"]), 4),
        "context_precision": round(float(scores["context_precision"]), 4),
        "context_recall":    round(float(scores["context_recall"]), 4),
        "avg_latency_s":     round(
            sum(r["latency_s"] for r in eval_rows) / len(eval_rows), 2
        ),
        "n_test_cases":      len(cases),
        "per_question":      eval_rows,
    }

    # ── Pretty-print results ──────────────────────────────────────────────────
    _print_results(results)

    # ── Optionally save to JSON ───────────────────────────────────────────────
    if output_path:
        os.makedirs(Path(output_path).parent, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results


def _print_results(results: dict) -> None:
    """Render a rich table of RAGAS scores with pass/fail indicators."""
    console = Console()

    table = Table(title="RAGAS Evaluation Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status", justify="center")

    thresholds = {
        "faithfulness":      0.88,
        "answer_relevancy":  0.85,
        "context_precision": 0.80,
        "context_recall":    0.75,
    }

    for metric, target in thresholds.items():
        score = results.get(metric, 0.0)
        passed = score >= target
        table.add_row(
            metric,
            f"{score:.4f}",
            f"{target:.2f}",
            "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]",
        )

    console.print(table)
    console.print(
        f"\n[dim]Test cases: {results['n_test_cases']} · "
        f"Avg latency: {results['avg_latency_s']}s[/dim]"
    )


def load_test_cases_from_file(path: str) -> list[dict]:
    """
    Load test cases from a JSON file.

    Expected format:
        [
            {"question": "...", "ground_truth": "..."},
            ...
        ]
    """
    with open(path) as f:
        cases = json.load(f)
    logger.info(f"Loaded {len(cases)} test cases from {path}")
    return cases


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline.")
    parser.add_argument(
        "--test-cases",
        default=None,
        help="Path to a JSON file with test cases (list of {question, ground_truth}).",
    )
    parser.add_argument(
        "--output",
        default="results/ragas_scores.json",
        help="Path to write the JSON results file.",
    )
    args = parser.parse_args()

    cases = None
    if args.test_cases:
        cases = load_test_cases_from_file(args.test_cases)

    run_evaluation(test_cases=cases, output_path=args.output)
