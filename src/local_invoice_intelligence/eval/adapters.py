from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from local_invoice_intelligence.extraction.fields import FieldDefinition


@dataclass(frozen=True)
class EvalDocument:
    doc_id: str
    pdf_path: Path
    ground_truth: dict[str, Any]


class DatasetAdapter:
    name = "base"

    def iter_documents(self, dataset_root: str | Path, fields: list[FieldDefinition]) -> Iterable[EvalDocument]:
        raise NotImplementedError


class DocileAdapter(DatasetAdapter):
    name = "docile"

    def iter_documents(self, dataset_root: str | Path, fields: list[FieldDefinition]) -> Iterable[EvalDocument]:
        from docile.dataset import Dataset

        root = Path(dataset_root)
        dataset = Dataset("val", str(root))
        requested = {field.name for field in fields}
        subset_path = root / "eval_subset_ids_sorted.json"
        doc_ids = json.loads(subset_path.read_text()) if subset_path.exists() else list(dataset.docids)
        for doc_id in doc_ids:
            doc = dataset[doc_id]
            ground_truth = {name: None for name in requested}
            for field in doc.annotation.fields:
                if field.fieldtype in requested:
                    ground_truth[field.fieldtype] = (
                        f"{ground_truth[field.fieldtype]} {field.text}"
                        if ground_truth[field.fieldtype]
                        else field.text
                    )
            yield EvalDocument(doc_id=doc_id, pdf_path=root / "pdfs" / f"{doc_id}.pdf", ground_truth=ground_truth)


class JsonAnnotationFolderAdapter(DatasetAdapter):
    """Adapter for benchmarks represented as pdfs/ plus annotations/*.json."""

    annotation_dir = "annotations"
    pdf_dir = "pdfs"

    def iter_documents(self, dataset_root: str | Path, fields: list[FieldDefinition]) -> Iterable[EvalDocument]:
        root = Path(dataset_root)
        requested = {field.name for field in fields}
        for pdf_path in sorted((root / self.pdf_dir).glob("*.pdf")):
            annotation_path = root / self.annotation_dir / f"{pdf_path.stem}.json"
            if not annotation_path.exists():
                continue
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "fields" in payload:
                payload = payload["fields"]
            ground_truth = {name: payload.get(name) for name in requested}
            yield EvalDocument(doc_id=pdf_path.stem, pdf_path=pdf_path, ground_truth=ground_truth)


class SroieAdapter(JsonAnnotationFolderAdapter):
    name = "sroie"


class KleisterAdapter(JsonAnnotationFolderAdapter):
    name = "kleister"


ADAPTERS: dict[str, DatasetAdapter] = {
    "docile": DocileAdapter(),
    "sroie": SroieAdapter(),
    "kleister": KleisterAdapter(),
}
