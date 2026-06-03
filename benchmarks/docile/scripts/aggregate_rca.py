from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rca_runner import Presence, classify_failure, score_field, success_threshold


DEFAULT_RCA_PATH = Path("results/rca_analysis_results.json")


def load_documents(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return {}, payload
    return payload.get("metadata", {}), payload.get("documents", [])


def pct(value: float, total: float) -> float:
    return 0.0 if total == 0 else (value / total) * 100


def short(value: Any, limit: int = 96) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def normalized_result(field_name: str, result: dict[str, Any], presence: dict[str, Any]) -> dict[str, Any]:
    updated = dict(result)
    ground_truth = updated.get("ground_truth")
    predicted = updated.get("predicted")
    score = score_field(field_name, ground_truth, predicted)
    threshold = success_threshold(field_name, 90.0)
    error_type = classify_failure(
        field_name,
        ground_truth,
        predicted,
        score,
        Presence(**presence),
        threshold,
    )
    updated["score"] = round(score, 2)
    updated["success"] = error_type is None
    updated["error_type"] = error_type
    return updated


def collect_stats(documents: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    examples: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for doc in documents:
        if doc.get("pipeline_error"):
            continue
        presence = doc.get("presence", {})
        for model_label, model_record in doc.get("models", {}).items():
            model_stats = stats.setdefault(
                model_label,
                {
                    "fields": {},
                    "model_errors": 0,
                },
            )
            if model_record.get("model_error"):
                model_stats["model_errors"] += 1
                continue

            for field_name, stored_result in model_record.get("fields", {}).items():
                field_presence = presence.get(field_name, {})
                result = normalized_result(field_name, stored_result, field_presence)
                gt = result.get("ground_truth")
                if not gt:
                    continue

                field_stats = model_stats["fields"].setdefault(
                    field_name,
                    {
                        "count": 0,
                        "success": 0,
                        "score_sum": 0.0,
                        "presence": Counter(),
                        "errors": Counter(),
                    },
                )
                field_stats["count"] += 1
                field_stats["score_sum"] += float(result.get("score") or 0.0)
                field_stats["presence"][presence.get(field_name, {}).get("status", "unknown")] += 1

                if result.get("success"):
                    field_stats["success"] += 1
                    continue

                error_type = result.get("error_type") or "UNKNOWN"
                field_stats["errors"][error_type] += 1
                key = (model_label, field_name, error_type)
                if len(examples[key]) < 3:
                    examples[key].append(
                        {
                            "doc_id": doc.get("doc_id"),
                            "ground_truth": gt,
                            "predicted": result.get("predicted"),
                            "score": result.get("score"),
                            "presence": field_presence,
                        }
                    )

    return {"stats": stats, "examples": examples}


def print_summary(metadata: dict[str, Any], documents: list[dict[str, Any]], max_examples: int):
    collected = collect_stats(documents)
    stats = collected["stats"]
    examples = collected["examples"]

    print(f"\nROOT CAUSE ANALYSIS SUMMARY ({len(documents)} documents)")
    if metadata:
        print(f"Models: {metadata.get('models', {})}")
        summary = metadata.get("summary")
        if summary:
            print(f"Latency summary: {summary}")
    print("=" * 100)

    for model_label, model_stats in sorted(stats.items()):
        print(f"\nMODEL: {model_label}")
        if model_stats["model_errors"]:
            print(f"Model-level errors: {model_stats['model_errors']}")
        print("-" * 100)
        print(
            f"{'FIELD':<22} | {'AVG SCORE':>9} | {'SUCCESS':>9} | "
            f"{'PERCEPTION':>10} | {'OCR':>7} | {'CTX':>7} | {'REASON':>8} | {'FORMAT':>8} | {'EVAL':>7}"
        )
        print("-" * 100)

        for field_name, field_stats in sorted(model_stats["fields"].items()):
            count = field_stats["count"]
            errors = field_stats["errors"]
            avg_score = field_stats["score_sum"] / count if count else 0.0
            print(
                f"{field_name:<22} | "
                f"{avg_score:>8.2f}% | "
                f"{pct(field_stats['success'], count):>8.2f}% | "
                f"{errors['PERCEPTION_MISS']:>10} | "
                f"{errors['OCR_NOISE']:>7} | "
                f"{errors['PROMPT_CONTEXT_MISS']:>7} | "
                f"{errors['REASONING_MISS']:>8} | "
                f"{errors['SCHEMA_OR_FORMAT_MISS']:>8} | "
                f"{errors['EVAL_NORMALIZATION_MISS']:>7}"
            )

        print("\nPresence distribution:")
        for field_name, field_stats in sorted(model_stats["fields"].items()):
            presence_bits = ", ".join(
                f"{name}={count}" for name, count in sorted(field_stats["presence"].items())
            )
            print(f"  {field_name}: {presence_bits}")

    print("\nEXAMPLES")
    print("=" * 100)
    for (model_label, field_name, error_type), items in sorted(examples.items()):
        print(f"\n{model_label} / {field_name} / {error_type}")
        for item in items[:max_examples]:
            presence = item.get("presence", {})
            print(
                f"- {item['doc_id']} score={item['score']} presence={presence.get('status')} "
                f"page={presence.get('page')}"
            )
            print(f"  gt  : {short(item.get('ground_truth'))}")
            print(f"  pred: {short(item.get('predicted'))}")
            snippet = short(presence.get("snippet"), limit=180)
            if snippet:
                print(f"  near: {snippet}")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate RCA output.")
    parser.add_argument("--report", default=str(DEFAULT_RCA_PATH))
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"File not found: {report_path}")
        return

    metadata, documents = load_documents(report_path)
    print_summary(metadata, documents, max_examples=args.examples)


if __name__ == "__main__":
    main()
