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

from observability import (
    get_current_trace_id,
    init_tracing,
    is_tracing_active,
    score_trace,
    shutdown_tracing,
    traced,
    update_current_observation,
)


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


# ── Single-case runner ────────────────────────────────────────────────────────
@traced(name="ragas_eval_case")
def _answer_one_case(engine, question: str) -> tuple[str, list[str], Optional[str]]:
    """
    Run one test case through the pipeline inside its own trace.

    The trace ID is captured here, while the span is still in scope, so the
    RAGAS scores computed later can be attached back to the exact trace whose
    retrieval and synthesis produced this answer.

    Returns (answer, contexts, trace_id). trace_id is None when tracing is off.
    """
    response = engine.query(question)
    answer = str(response)
    contexts = [node.node.text for node in response.source_nodes]

    update_current_observation(
        input=question,
        output=answer,
        n_contexts=len(contexts),
    )

    return answer, contexts, get_current_trace_id()


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

    # This is a CLI entry point, so it doesn't go through the FastAPI lifespan
    # that normally installs tracing. Set it up here so eval runs produce the
    # same full-pipeline traces a live request would. No-op unless enabled.
    init_tracing()

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
            answer, contexts, trace_id = _answer_one_case(engine, question)
            latency = time.perf_counter() - start

            eval_rows.append({
                "question":     question,
                "answer":       answer,
                "contexts":     contexts,
                "ground_truth": ground_truth,
                "latency_s":    round(latency, 2),
                "trace_id":     trace_id,
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
                "trace_id":     None,
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

    # ── Attach scores back to their traces ────────────────────────────────────
    _attach_scores_to_traces(scores, eval_rows)

    # ── Pretty-print results ──────────────────────────────────────────────────
    _print_results(results)

    # ── Optionally save to JSON ───────────────────────────────────────────────
    if output_path:
        os.makedirs(Path(output_path).parent, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return results


_RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _attach_scores_to_traces(scores, eval_rows: list[dict]) -> None:
    """
    Push each case's RAGAS scores onto the trace that produced its answer.

    This is the payoff of tracing the eval run: instead of a bare number in a
    JSON file, a low faithfulness score becomes a clickable trace showing the
    retrieved chunks, what the reranker kept or dropped, and the exact prompt
    the synthesizer built. Answers "why is this score bad?", not just "it is".

    No-op when tracing is off. Never raises — a scoring failure must not lose
    an eval run that has already spent real API budget.
    """
    if not is_tracing_active():
        return

    # RAGAS returns per-question score lists alongside the aggregates; fall
    # back to skipping rather than guessing if the shape isn't what we expect.
    try:
        per_question = scores.to_pandas()
    except Exception as e:
        logger.warning(f"Could not read per-question RAGAS scores ({e}) — traces left unscored.")
        return

    attached = 0
    for i, row in enumerate(eval_rows):
        trace_id = row.get("trace_id")
        if not trace_id or i >= len(per_question):
            continue

        for metric in _RAGAS_METRICS:
            if metric not in per_question.columns:
                continue

            value = per_question[metric].iloc[i]
            # RAGAS emits NaN when a metric can't be computed for a case.
            if value is None or value != value:
                continue

            score_trace(
                trace_id=trace_id,
                name=metric,
                value=float(value),
                comment=f"RAGAS · {row['question'][:100]}",
            )
        attached += 1

    logger.info(f"Attached RAGAS scores to {attached}/{len(eval_rows)} traces.")
    shutdown_tracing()


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
