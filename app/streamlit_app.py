from __future__ import annotations

import json
import tempfile
from io import StringIO
from pathlib import Path
import csv

import streamlit as st
import yaml

from local_invoice_intelligence.extraction.fields import FieldDefinition
from local_invoice_intelligence.extraction.pipeline import extract_document


DEFAULT_FIELDS = """fields:
  - name: vendor_name
    description: The supplier, seller, or billing organization name.
    type: string
    required: true
  - name: invoice_number
    description: The invoice identifier or document number.
    type: string
    required: false
  - name: issue_date
    description: The date the document was issued or created.
    type: date
    required: false
  - name: total_due
    description: The final amount due, grand total, or balance payable.
    type: money
    required: false
"""


def _parse_fields(text: str) -> list[FieldDefinition]:
    payload = yaml.safe_load(text) or {}
    return [FieldDefinition.from_mapping(item) for item in payload.get("fields", payload)]


st.set_page_config(page_title="Local PDF Extraction", layout="wide")
st.title("Local PDF Extraction")

uploaded = st.file_uploader("PDF", type=["pdf"])
model = st.text_input("Ollama model", value="qwen3:14b")
field_text = st.text_area("Fields", value=DEFAULT_FIELDS, height=260)

if st.button("Run extraction", type="primary", disabled=uploaded is None):
    fields = _parse_fields(field_text)
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / uploaded.name
        pdf_path.write_bytes(uploaded.getvalue())
        with st.spinner("Extracting locally..."):
            result = extract_document(pdf_path=pdf_path, fields=fields, model=model, provider="ollama")
    st.subheader("JSON")
    st.json(result)
    st.subheader("Table")
    row = {"source_file": uploaded.name, **result["fields"], **result["metadata"]}
    st.dataframe([row])
    csv_buffer = StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    st.download_button(
        "Download JSON",
        data=json.dumps(result, indent=2),
        file_name=f"{Path(uploaded.name).stem}.json",
        mime="application/json",
    )
    st.download_button(
        "Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"{Path(uploaded.name).stem}.csv",
        mime="text/csv",
    )
