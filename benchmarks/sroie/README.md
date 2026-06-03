# SROIE Adapter

The general evaluator expects a normalized local layout:

```text
dataset-root/
  pdfs/
    sample.pdf
  annotations/
    sample.json
```

Each annotation JSON can either be a flat object keyed by requested field name or an object with a `fields` object.

```json
{
  "fields": {
    "company": "Example Store",
    "date": "2019-01-01",
    "total": "12.30"
  }
}
```

Run with field definitions that match the annotation keys:

```bash
uv run lii eval --dataset sroie --dataset-root /path/to/sroie-normalized --fields configs/example_fields.yaml --out results/sroie_report.json
```
