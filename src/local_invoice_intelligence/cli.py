from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_invoice_intelligence.batch.runner import extract_folder
from local_invoice_intelligence.config import MODEL_PROVIDER, OCR_BACKEND, TEXT_MODEL
from local_invoice_intelligence.extraction.fields import load_field_definitions
from local_invoice_intelligence.extraction.pipeline import extract_document


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=TEXT_MODEL, help="Model name. Defaults to TEXT_MODEL or qwen3:14b.")
    parser.add_argument("--provider", choices=["ollama", "openai"], default=MODEL_PROVIDER)
    parser.add_argument("--ocr-backend", choices=["tesseract", "rapidocr"], default=OCR_BACKEND)
    parser.add_argument(
        "--ollama-thinking",
        choices=["auto", "false", "true", "low", "medium", "high"],
        default=None,
        help="Ollama thinking control for compatible local models.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local private PDF extraction toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    one = subparsers.add_parser("extract-one", help="Extract fields from one PDF into JSON.")
    one.add_argument("--pdf", required=True)
    one.add_argument("--fields", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--include-transcript", action="store_true")
    _add_model_args(one)

    folder = subparsers.add_parser("extract-folder", help="Extract fields from a folder of PDFs into Excel.")
    folder.add_argument("--input-dir", required=True)
    folder.add_argument("--fields", required=True)
    folder.add_argument("--out", required=True)
    folder.add_argument("--artifacts-dir", default=None)
    _add_model_args(folder)

    eval_cmd = subparsers.add_parser("eval", help="Run a benchmark adapter.")
    eval_cmd.add_argument("--dataset", choices=["docile", "sroie", "kleister"], required=True)
    eval_cmd.add_argument("--fields", required=True, help="Field definition YAML/JSON for the evaluation.")
    eval_cmd.add_argument("--dataset-root", required=True)
    eval_cmd.add_argument("--out", required=True)
    _add_model_args(eval_cmd)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fields = load_field_definitions(args.fields)
    if args.command == "extract-one":
        result = extract_document(
            pdf_path=args.pdf,
            fields=fields,
            model=args.model,
            provider=args.provider,
            ocr_backend=args.ocr_backend,
            ollama_thinking=args.ollama_thinking,
            include_transcript=args.include_transcript,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
        return
    if args.command == "extract-folder":
        extract_folder(
            input_dir=args.input_dir,
            fields=fields,
            out_path=args.out,
            artifacts_dir=args.artifacts_dir,
            model=args.model,
            provider=args.provider,
            ocr_backend=args.ocr_backend,
            ollama_thinking=args.ollama_thinking,
        )
        print(f"Wrote {args.out}")
        return
    if args.command == "eval":
        from local_invoice_intelligence.eval.runner import run_eval

        report = run_eval(
            dataset=args.dataset,
            dataset_root=args.dataset_root,
            fields=fields,
            model=args.model,
            provider=args.provider,
            ocr_backend=args.ocr_backend,
            ollama_thinking=args.ollama_thinking,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
