"""Command-line entry point: `python -m ai_ops.evaluation`.

Why: One command runs everything, prints a report, stores it with its date and configuration, and
sets an exit code that CI can act on.

Exit codes: 0 means passed, 1 means a score is below its minimum, and 2 means the run could not
complete. A broken setup is kept apart from a real regression.

Used by: people, and the evaluation step in `.github/workflows/ci.yml`.
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_ops.config import ConfigError, Settings
from ai_ops.evaluation.golden import GoldenError, load_golden
from ai_ops.evaluation.metrics import check_thresholds
from ai_ops.evaluation.runner import run_evaluation
from ai_ops.llm.provider import LLMError
from ai_ops.logging import configure_logging
from ai_ops.retrieval.documents import load_documents

_SCORES = (
    "hit_rate",
    "precision",
    "recall",
    "context_fact_recall",
    "not_found_accuracy",
    "answer_fact_recall",
    "groundedness",
    "citation_rate",
    "injection_resistance",
)
_OTHER = ("latency_p50_ms", "latency_p95_ms", "input_tokens", "output_tokens", "estimated_cost")


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line.

    Why: The defaults make the common case a single command. Retrieval settings come from the same
    environment variables as the API, so what is evaluated matches what would be deployed.

    Used by: `main`.

    Args:
        argv: The arguments. None means the real command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline on the golden set.")
    parser.add_argument("--golden", type=Path, default=Path("data/evaluation/golden.json"))
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("data/evaluation/thresholds.json"),
        help="JSON file with minimum scores for each provider.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation-runs"))
    parser.add_argument("--label", default="run", help="Short name stored with this run.")
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    return parser.parse_args(argv)


def _format_report(report: dict[str, Any]) -> str:
    """Format the report for the terminal.

    Why: "not measured" is printed in words, so nobody can mistake a missing number for a score.

    Used by: `main`.

    Args:
        report: The report.

    Returns:
        The text to print.
    """
    metadata = report["metadata"]
    metrics = report["metrics"]

    def line(name: str) -> str:
        """Format one metric."""
        value = metrics[name]
        shown = "not measured" if value is None else f"{value:.3f}".rstrip("0").rstrip(".")
        return f"  {name:<24}{shown}"

    settings = (
        f"chunk_size={metadata['chunk_size']} chunk_overlap={metadata['chunk_overlap']} "
        f"top_k={metadata['top_k']} min_score={metadata['min_score']}"
    )
    return "\n".join(
        [
            f"Evaluation '{metadata['label']}' on {metadata['date']}",
            f"Provider: {metadata['provider']}, {metadata['case_count']} cases",
            f"Settings: {settings}",
            "",
            "Scores (0 to 1)",
            *[line(name) for name in _SCORES],
            "",
            "Cost and speed",
            *[line(name) for name in _OTHER],
        ]
    )


def _store_report(report: dict[str, Any], directory: Path) -> Path:
    """Write the report to a new JSON file.

    Why: A stored run with its date and configuration is the evidence behind any quoted number. The
    time in the name keeps earlier runs from being overwritten.

    Used by: `main`.

    Args:
        report: The report.
        directory: Where to write it. It is created if missing.

    Returns:
        The new file's path.

    Side effects:
        Creates a file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = directory / f"{stamp}_{report['metadata']['label']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation and check the thresholds.

    Why: Every expected failure ends in a short message and exit code 2, so a broken setup cannot be
    mistaken for a regression.

    Used by: the `__main__` block below.

    Args:
        argv: The arguments. None means the real command line.

    Returns:
        0 if passed, 1 if a score is too low, 2 if the run could not complete.

    Side effects:
        Reads files, calls the model provider, prints the report, and stores it.
    """
    arguments = _parse_arguments(argv)
    try:
        settings = Settings.from_env()
        configure_logging(settings.log_level)
        documents = load_documents(settings.knowledge_dir)
        golden = load_golden(arguments.golden, {document.id for document in documents})
        report = asyncio.run(
            run_evaluation(
                settings,
                golden,
                documents,
                label=arguments.label,
                input_price_per_million=arguments.input_price_per_million,
                output_price_per_million=arguments.output_price_per_million,
            )
        )
        thresholds = (
            json.loads(arguments.thresholds.read_text(encoding="utf-8"))
            if arguments.thresholds.exists()
            else {}
        )
    except (ConfigError, GoldenError, LLMError, ValueError) as exc:
        print(f"Evaluation could not complete: {exc}", file=sys.stderr)
        return 2

    print(_format_report(report))
    print(f"\nStored: {_store_report(report, arguments.output_dir)}")

    failures = check_thresholds(report["metrics"], thresholds.get(settings.llm_provider, {}))
    if failures:
        print("\nFAILED")
        print("\n".join(f"  {failure}" for failure in failures))
        return 1
    print("\nPASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
