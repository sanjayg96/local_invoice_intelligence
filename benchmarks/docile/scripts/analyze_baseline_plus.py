import argparse
import json
from copy import deepcopy

from evaluate_metrics import FIELDS, score_record, summarize_scores


def _score_report(data):
    scores = {field: [] for field in FIELDS}
    for item in data:
        for field, score in score_record(item).items():
            scores[field].append(score)
    return summarize_scores(scores)


def _with_shadow_applied(data):
    shadow_data = deepcopy(data)
    for item in shadow_data:
        predicted = item.get("predicted", {})
        for field_name, record in item.get("field_results", {}).items():
            decision = record.get("shadow_rescue_decision") or {}
            candidate = record.get("shadow_rescue_candidate") or {}
            if decision.get("accepted") and candidate.get("value"):
                predicted[field_name] = candidate["value"]
                if field_name == "amount_total_gross":
                    predicted["invoice_total"] = candidate["value"]
    return shadow_data


def _count_values(data, metadata_key):
    counts = {}
    for item in data:
        for field_name in item.get("pipeline_metadata", {}).get(metadata_key, []):
            counts[field_name] = counts.get(field_name, 0) + 1
    return counts


def _decision_counts(data, metadata_key):
    counts = {}
    for item in data:
        decisions = item.get("pipeline_metadata", {}).get(metadata_key, {})
        for field_name, decision in decisions.items():
            key = (field_name, str(bool(decision.get("accepted"))), decision.get("reason", ""))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _print_summary(title, summary):
    print(title)
    for field_name, score in summary["fields"].items():
        print(f"  {field_name:<22}: {score:.2f}%")
    print(f"  {'total_average':<22}: {summary['total_average']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline-plus rescue and shadow rescue behavior.")
    parser.add_argument("--report", required=True, help="Path to a baseline-plus evaluation report.")
    args = parser.parse_args()

    with open(args.report, "r") as f:
        data = json.load(f)

    _print_summary("Actual predictions", _score_report(data))
    print()
    _print_summary("If accepted shadow rescues were applied", _score_report(_with_shadow_applied(data)))

    print("\nField counts")
    print(f"  weak_fields   : {_count_values(data, 'weak_fields')}")
    print(f"  rescue_fields : {_count_values(data, 'rescue_fields')}")
    print(f"  shadow_fields : {_count_values(data, 'shadow_fields')}")

    print("\nRescue decisions")
    for key, count in sorted(_decision_counts(data, "rescue_decisions").items()):
        print(f"  {count:>3}  {key[0]}\taccepted={key[1]}\t{key[2]}")

    print("\nShadow decisions")
    for key, count in sorted(_decision_counts(data, "shadow_decisions").items()):
        print(f"  {count:>3}  {key[0]}\taccepted={key[1]}\t{key[2]}")


if __name__ == "__main__":
    main()
