from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import Workbook

from local_invoice_intelligence.extraction.fields import FieldDefinition


def write_results_xlsx(results: list[dict[str, Any]], fields: list[FieldDefinition], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "extractions"
    headers = [
        "source_file",
        *[field.name for field in fields],
        "latency_s",
        "model",
        "warnings",
        "transcript_chars",
    ]
    ws.append(headers)
    for result in results:
        metadata = result.get("metadata", {})
        values = result.get("fields", {})
        ws.append(
            [
                metadata.get("source_file") or result.get("source_file"),
                *[values.get(field.name) for field in fields],
                metadata.get("latency_s"),
                metadata.get("model"),
                "; ".join(metadata.get("warnings") or []),
                metadata.get("transcript_chars"),
            ]
        )
    for column in ws.columns:
        max_length = max(len(str(cell.value or "")) for cell in column)
        ws.column_dimensions[column[0].column_letter].width = min(max(max_length + 2, 12), 60)
    wb.save(out_path)
