from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import ollama
from dateutil import parser as date_parser
from docile.dataset import Dataset
from rapidfuzz import fuzz
from tqdm import tqdm

from config import DOCILE_DATASET_PATH, SUBSET_JSON_PATH, TEXT_MODEL, ENABLE_THINKING
from extractor import extract_text_hybrid
from ollama_controls import maybe_add_thinking_directive, primary_chat_controls
from schema import InvoiceExtractionBase, InvoiceExtractionWithReasoning


DEFAULT_MODELS = {
    "qwen3_14b": TEXT_MODEL,
    "3b": "llama3.2:3b",
    "8b": "llama3.1:8b",
}
DEFAULT_FIELDS = ("vendor_name", "vendor_address", "amount_total_gross", "date_issue")
DEFAULT_OUTPUT = Path("results/rca_analysis_results.json")
PROMPT_CONTEXT_CHAR_BUDGET = 24000
MONTH_RE = r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
DATE_LIKE_RE = re.compile(
    rf"""
    (?<!\w)
    (?:
        \d{{4}}[/-]\d{{1,2}}[/-]\d{{1,2}}
        |
        \d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}
        |
        (?:{MONTH_RE})[a-z]*\.?\s*\d{{1,2}},?\s*\d{{2,4}}
        |
        (?:{MONTH_RE})[a-z]*\.?\s*\d{{1,2}}[/-]\d{{2,4}}
        |
        \d{{1,2}}\s*(?:{MONTH_RE})[a-z]*\.?,?\s*\d{{2,4}}
    )
    (?!\w)
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class Presence:
    status: str
    exact: bool
    best_score: float
    snippet: str
    char_start: Optional[int]
    page: Optional[int]
    prompt_context_risk: bool


def normalize_loose(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_text(text: Any) -> str:
    text = normalize_loose(text).lower()
    return re.sub(r"[^\w\s]", "", text).strip()


def normalize_compact(text: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", str(text or "")).lower()


def deduplicate_docile_artifacts(text: str) -> str:
    words = normalize_loose(text).split()
    if len(words) < 2:
        return normalize_loose(text)

    for pattern_len in range(1, len(words) // 2 + 1):
        if len(words) % pattern_len != 0:
            continue
        pattern = words[:pattern_len]
        if pattern * (len(words) // pattern_len) == words:
            return " ".join(pattern)
    return normalize_loose(text)


def parse_first_float(text: Any) -> Optional[float]:
    if not text:
        return None
    cleaned = str(text).replace(",", "")
    decimal_matches = re.findall(r"\d+\.\d+", cleaned)
    if decimal_matches:
        return float(decimal_matches[-1])

    all_matches = re.findall(r"\d+", cleaned)
    if all_matches:
        return float(max(all_matches, key=int))
    return None


def _expand_compact_month(value: str) -> str:
    return re.sub(
        rf"\b({MONTH_RE})(\d{{1,2}})([/-]\d{{2,4}})\b",
        lambda match: f"{match.group(1)} {match.group(2)}{match.group(3)}",
        value,
        flags=re.IGNORECASE,
    )


def parse_date(text: Any):
    if not text:
        return None
    cleaned = re.sub(r"\b(?:EDT|EST|ET|CDT|CST|CT|MDT|MST|MT|PDT|PST|PT|UTC|GMT)\b", " ", str(text))
    try:
        parsed = date_parser.parse(_expand_compact_month(cleaned), fuzzy=True).date()
        if parsed.year < 1900 or parsed.year > 2100:
            return None
        return parsed
    except Exception:
        return None


def candidate_dates(text: Any) -> list:
    raw = str(text or "")
    candidates = []
    seen = set()
    for item in [raw, *DATE_LIKE_RE.findall(raw), *re.split(r"\s+", raw)]:
        parsed = parse_date(item)
        if parsed and parsed not in seen:
            candidates.append(parsed)
            seen.add(parsed)
    return candidates


def candidate_amounts(text: Any) -> list[float]:
    raw = str(text or "").replace(",", "")
    values = []
    for match in re.findall(r"[-+]?\$?\s*\d+(?:\.\d+)?", raw):
        value = parse_first_float(match)
        if value is not None:
            values.append(value)
    parsed = parse_first_float(raw)
    if parsed is not None:
        values.append(parsed)
    return values


def score_field(field_name: str, ground_truth: Any, prediction: Any) -> float:
    if not ground_truth and not prediction:
        return 100.0
    if not ground_truth or not prediction:
        return 0.0

    if field_name == "amount_total_gross":
        gt_num = parse_first_float(ground_truth)
        pred_nums = candidate_amounts(prediction)
        if gt_num is None and not pred_nums:
            return 100.0
        if gt_num is None or not pred_nums:
            return 0.0
        return 100.0 if any(gt_num == pred_num for pred_num in pred_nums) else 0.0

    if field_name == "date_issue":
        gt_dates = candidate_dates(ground_truth)
        pred_dates = candidate_dates(prediction)
        if gt_dates and pred_dates:
            return 100.0 if any(gt_date == pred_date for gt_date in gt_dates for pred_date in pred_dates) else 0.0
        return 100.0 if normalize_compact(ground_truth) == normalize_compact(prediction) else 0.0

    return float(fuzz.token_set_ratio(normalize_text(prediction), normalize_text(ground_truth)))


def success_threshold(field_name: str, text_threshold: float) -> float:
    if field_name in {"amount_total_gross", "date_issue"}:
        return 100.0
    return text_threshold


def transcript_lines(transcription: str) -> list[tuple[Optional[int], int, str]]:
    lines = []
    current_page: Optional[int] = None
    offset = 0

    for raw_line in transcription.splitlines(keepends=True):
        line = raw_line.rstrip("\n")
        page_match = re.search(r"START OF PAGE\s+(\d+)", line)
        if page_match:
            current_page = int(page_match.group(1))
        if line.strip():
            lines.append((current_page, offset, line.strip()))
        offset += len(raw_line)
    return lines


def _windowed_lines(lines: list[tuple[Optional[int], int, str]], radius: int = 2):
    for idx, (page, char_start, _) in enumerate(lines):
        start = max(0, idx - radius)
        end = min(len(lines), idx + radius + 1)
        page_window = lines[start:end]
        text = "\n".join(line for _, _, line in page_window)
        yield page, char_start, text


def find_presence(ground_truth: Any, transcription: str) -> Presence:
    gt = deduplicate_docile_artifacts(str(ground_truth or ""))
    compact_gt = normalize_compact(gt)
    if not compact_gt:
        return Presence("empty_ground_truth", True, 100.0, "", None, None, False)

    compact_transcript = normalize_compact(transcription)
    exact = compact_gt in compact_transcript

    best_score = 0.0
    best_snippet = ""
    best_start: Optional[int] = None
    best_page: Optional[int] = None
    for page, char_start, window in _windowed_lines(transcript_lines(transcription)):
        score = max(
            fuzz.token_set_ratio(normalize_text(gt), normalize_text(window)),
            fuzz.partial_ratio(compact_gt, normalize_compact(window)),
        )
        if score > best_score:
            best_score = float(score)
            best_snippet = window
            best_start = char_start
            best_page = page

    if exact:
        status = "exact"
    elif best_score >= 85:
        status = "fuzzy"
    else:
        status = "absent"

    prompt_context_risk = bool(
        exact
        and best_start is not None
        and len(transcription) > PROMPT_CONTEXT_CHAR_BUDGET
        and best_start > PROMPT_CONTEXT_CHAR_BUDGET
    )
    return Presence(
        status=status,
        exact=exact,
        best_score=round(best_score, 2),
        snippet=best_snippet[:1600],
        char_start=best_start,
        page=best_page,
        prompt_context_risk=prompt_context_risk,
    )


def extract_ground_truth(doc) -> dict[str, Optional[str]]:
    gt_data: dict[str, Optional[str]] = {field: None for field in DEFAULT_FIELDS}
    for field in doc.annotation.fields:
        if field.fieldtype not in gt_data:
            continue
        if gt_data[field.fieldtype]:
            gt_data[field.fieldtype] += " " + field.text
        else:
            gt_data[field.fieldtype] = field.text
    return gt_data


def parse_models(raw_models: list[str]) -> dict[str, str]:
    models = {}
    for item in raw_models:
        if "=" in item:
            label, model = item.split("=", 1)
            models[label.strip()] = model.strip()
        else:
            label = item.replace(":", "_").replace("/", "_")
            models[label] = item
    return models


def extract_json(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def run_auditor(transcription: str, model_name: str) -> tuple[dict[str, Any], float, str]:
    schema = InvoiceExtractionWithReasoning if ENABLE_THINKING else InvoiceExtractionBase
    system_prompt = (
        "You are an expert financial data auditor. You will be provided with a complete, "
        "multi-page text extraction of an invoice. The text has been formatted to preserve "
        "the original physical layout of the document.\n\n"
        "Return only JSON matching the requested schema. Copy exact visible values when possible. "
        "For totals, choose the final total due or gross amount, not a subtotal, tax, unit price, "
        "or line-item amount."
    )
    if "reasoning_process" in getattr(schema, "model_fields", {}):
        system_prompt += (
            "\n\nCRITICAL INSTRUCTION FOR `reasoning_process`: This field is a brief scratchpad. "
            "Use it to verify the vendor, issue date, and final total before populating the JSON fields."
        )

    started = time.perf_counter()
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": maybe_add_thinking_directive(f"DOCUMENT TRANSCRIPTION:\n{transcription}"),
            },
        ],
        format=schema.model_json_schema(),
        **primary_chat_controls(),
        keep_alive="10m",
    )
    latency_s = time.perf_counter() - started
    content = response["message"]["content"]
    return extract_json(content), latency_s, content


def prediction_value(prediction: dict[str, Any], field_name: str) -> Any:
    if field_name == "amount_total_gross":
        return prediction.get("amount_total_gross") or prediction.get("invoice_total")
    return prediction.get(field_name)


def classify_failure(
    field_name: str,
    ground_truth: Any,
    prediction: Any,
    score: float,
    presence: Presence,
    threshold: float,
) -> Optional[str]:
    if score >= threshold:
        return None
    if not ground_truth:
        return "FIELD_ABSENT_IN_GT"

    if field_name == "amount_total_gross":
        gt_num = parse_first_float(ground_truth)
        if gt_num is not None and any(gt_num == value for value in candidate_amounts(prediction)):
            return "EVAL_NORMALIZATION_MISS"
    elif field_name == "date_issue":
        gt_dates = candidate_dates(ground_truth)
        pred_dates = candidate_dates(prediction)
        if gt_dates and pred_dates and any(gt_date == pred_date for gt_date in gt_dates for pred_date in pred_dates):
            return "EVAL_NORMALIZATION_MISS"
    else:
        compact_gt = normalize_compact(ground_truth)
        compact_pred = normalize_compact(prediction)
        shorter = min(len(compact_gt), len(compact_pred))
        longer = max(len(compact_gt), len(compact_pred))
        if (
            shorter >= 8
            and longer
            and shorter / longer >= 0.55
            and (compact_gt in compact_pred or compact_pred in compact_gt)
        ):
            return "EVAL_NORMALIZATION_MISS"

    if presence.status == "absent":
        return "PERCEPTION_MISS"
    if presence.status == "fuzzy":
        return "OCR_NOISE"
    if presence.prompt_context_risk:
        return "PROMPT_CONTEXT_MISS"

    if field_name == "amount_total_gross" and prediction and parse_first_float(prediction) is None:
        return "SCHEMA_OR_FORMAT_MISS"
    if field_name == "date_issue" and prediction and parse_date(prediction) is None:
        return "SCHEMA_OR_FORMAT_MISS"

    return "REASONING_MISS"


def page_count_from_transcript(transcription: str) -> int:
    pages = re.findall(r"START OF PAGE\s+(\d+)", transcription)
    return max((int(page) for page in pages), default=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline RCA for local invoice extraction.")
    parser.add_argument("--dataset-path", default=DOCILE_DATASET_PATH)
    parser.add_argument("--subset", default=str(SUBSET_JSON_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=[f"{k}={v}" for k, v in DEFAULT_MODELS.items()])
    parser.add_argument("--text-success-threshold", type=float, default=90.0)
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--cooldown-batch-size", type=int, default=25)
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Do not store full transcripts in the RCA JSON. Snippets and stats are still stored.",
    )
    return parser.parse_args()


def summarize_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    model_latencies: dict[str, list[float]] = {}
    parse_latencies = []
    for record in records:
        if record.get("parse_latency_s") is not None:
            parse_latencies.append(float(record["parse_latency_s"]))
        for label, model_record in record.get("models", {}).items():
            if model_record.get("latency_s") is not None:
                model_latencies.setdefault(label, []).append(float(model_record["latency_s"]))

    return {
        "documents": len(records),
        "parse_latency_avg_s": round(statistics.mean(parse_latencies), 3) if parse_latencies else None,
        "model_latency_avg_s": {
            label: round(statistics.mean(values), 3)
            for label, values in sorted(model_latencies.items())
            if values
        },
    }


def main():
    args = parse_args()
    models = parse_models(args.models)

    print(f"Loading DocILE dataset from {args.dataset_path}...")
    dataset = Dataset("val", args.dataset_path)
    with open(args.subset, "r") as f:
        subset_ids = json.load(f)

    selected_ids = subset_ids[args.start :]
    if args.limit is not None:
        selected_ids = selected_ids[: args.limit]

    print(f"Running RCA on {len(selected_ids)} documents.")
    print(f"Models: {models}")
    records = []

    for idx, doc_id in enumerate(tqdm(selected_ids, desc="RCA baseline")):
        doc = dataset[doc_id]
        ground_truth = extract_ground_truth(doc)
        pdf_path = os.path.join(args.dataset_path, "pdfs", f"{doc_id}.pdf")

        record: dict[str, Any] = {
            "doc_id": doc_id,
            "pdf_path": pdf_path,
            "ground_truth": ground_truth,
            "models": {},
        }

        try:
            parse_started = time.perf_counter()
            transcription = extract_text_hybrid(pdf_path)
            parse_latency_s = time.perf_counter() - parse_started

            presences = {
                field_name: asdict(find_presence(value, transcription))
                for field_name, value in ground_truth.items()
            }
            record.update(
                {
                    "parse_latency_s": round(parse_latency_s, 4),
                    "transcript_stats": {
                        "characters": len(transcription),
                        "lines": len(transcription.splitlines()),
                        "pages": page_count_from_transcript(transcription),
                    },
                    "transcript": None if args.no_transcript else transcription,
                    "presence": presences,
                }
            )

            for label, model_name in models.items():
                model_record: dict[str, Any] = {"model": model_name, "fields": {}}
                try:
                    prediction, latency_s, raw_content = run_auditor(transcription, model_name)
                    model_record.update(
                        {
                            "latency_s": round(latency_s, 4),
                            "prediction": prediction,
                            "raw_response": raw_content,
                        }
                    )

                    for field_name, gt_value in ground_truth.items():
                        pred_value = prediction_value(prediction, field_name)
                        score = score_field(field_name, gt_value, pred_value)
                        threshold = success_threshold(field_name, args.text_success_threshold)
                        error_type = classify_failure(
                            field_name,
                            gt_value,
                            pred_value,
                            score,
                            Presence(**presences[field_name]),
                            threshold,
                        )
                        model_record["fields"][field_name] = {
                            "ground_truth": gt_value,
                            "predicted": pred_value,
                            "score": round(score, 2),
                            "success": error_type is None,
                            "error_type": error_type,
                        }
                except Exception as exc:
                    model_record["model_error"] = str(exc)
                record["models"][label] = model_record

        except Exception as exc:
            record["pipeline_error"] = str(exc)

        records.append(record)

        if (
            args.cooldown_seconds > 0
            and args.cooldown_batch_size > 0
            and (idx + 1) % args.cooldown_batch_size == 0
            and idx < len(selected_ids) - 1
        ):
            print(f"\nCooling down for {args.cooldown_seconds}s...")
            time.sleep(args.cooldown_seconds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "dataset_path": args.dataset_path,
            "subset": args.subset,
            "models": models,
            "fields": list(DEFAULT_FIELDS),
            "text_success_threshold": args.text_success_threshold,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "summary": summarize_run(records),
        },
        "documents": records,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nRCA complete. Saved {len(records)} documents to {output_path}")
    print(json.dumps(payload["metadata"]["summary"], indent=2))


if __name__ == "__main__":
    main()
