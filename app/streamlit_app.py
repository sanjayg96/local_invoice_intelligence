from __future__ import annotations

import json
import tempfile
from io import StringIO
from pathlib import Path
import csv

import streamlit as st

from local_invoice_intelligence.extraction.fields import FieldDefinition
from local_invoice_intelligence.extraction.pipeline import extract_document


MODEL_OPTIONS = {
    "Qwen3 14B": "qwen3:14b",
    "Llama 3.1 8B": "llama3.1:8b",
    "Llama 3.2 3B": "llama3.2:3b",
}

FIELD_TYPE_OPTIONS = {
    "Text": "string",
    "Date": "date",
    "Money": "money",
    "Number": "number",
    "Yes / No": "boolean",
}

DEFAULT_FIELDS = [
    {
        "name": "vendor_name",
        "description": "The supplier, seller, or billing organization name.",
        "type": "string",
    },
    {
        "name": "invoice_number",
        "description": "The invoice identifier or document number.",
        "type": "string",
    },
    {
        "name": "issue_date",
        "description": "The date the document was issued or created.",
        "type": "date",
    },
    {
        "name": "total_due",
        "description": "The final amount due, grand total, or balance payable.",
        "type": "money",
    },
]


def _init_fields() -> None:
    if "fields" not in st.session_state:
        st.session_state.fields = [dict(field) for field in DEFAULT_FIELDS]


def _add_field() -> None:
    st.session_state.fields.append(
        {
            "name": "",
            "description": "",
            "type": "string",
        }
    )


def _remove_field(index: int) -> None:
    st.session_state.fields.pop(index)


def _type_label(value: str) -> str:
    for label, option_value in FIELD_TYPE_OPTIONS.items():
        if option_value == value:
            return label
    return "Text"


def _field_definitions() -> list[FieldDefinition]:
    fields = []
    for field in st.session_state.fields:
        name = str(field.get("name") or "").strip()
        description = str(field.get("description") or "").strip()
        if not name or not description:
            continue
        fields.append(
            FieldDefinition(
                name=name,
                description=description,
                type=str(field.get("type") or "string"),
                required=False,
            )
        )
    return fields


st.set_page_config(page_title="Local PDF Extraction", layout="wide")
st.title("Local PDF Extraction")
_init_fields()

uploaded = st.file_uploader("PDF", type=["pdf"])
model_label = st.selectbox("Ollama model", options=list(MODEL_OPTIONS), index=0)
model = MODEL_OPTIONS[model_label]

st.subheader("Fields")
for index, field in enumerate(st.session_state.fields):
    with st.container(border=True):
        top_left, top_right = st.columns([10, 1])
        with top_left:
            st.markdown(f"**Field {index + 1}**")
        with top_right:
            st.button(
                "Remove",
                key=f"remove_{index}",
                on_click=_remove_field,
                args=(index,),
                disabled=len(st.session_state.fields) <= 1,
            )

        name_col, type_col = st.columns([2, 1])
        with name_col:
            st.session_state.fields[index]["name"] = st.text_input(
                "Name",
                value=field.get("name", ""),
                key=f"name_{index}",
                placeholder="example: purchase_order_number",
            )
        with type_col:
            current_label = _type_label(str(field.get("type") or "string"))
            selected_label = st.selectbox(
                "Kind of value",
                options=list(FIELD_TYPE_OPTIONS),
                index=list(FIELD_TYPE_OPTIONS).index(current_label),
                key=f"type_{index}",
            )
            st.session_state.fields[index]["type"] = FIELD_TYPE_OPTIONS[selected_label]

        st.session_state.fields[index]["description"] = st.text_area(
            "What should be extracted?",
            value=field.get("description", ""),
            key=f"description_{index}",
            height=80,
            placeholder="Describe where this value appears and what it means.",
        )

st.button("Add field", on_click=_add_field)

if st.button("Run extraction", type="primary", disabled=uploaded is None):
    fields = _field_definitions()
    field_names = [field.name for field in fields]
    duplicate_names = sorted({name for name in field_names if field_names.count(name) > 1})
    if not fields:
        st.error("Add at least one field with a name and description.")
        st.stop()
    if duplicate_names:
        st.error(f"Field names must be unique: {', '.join(duplicate_names)}")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / uploaded.name
        pdf_path.write_bytes(uploaded.getvalue())
        with st.spinner("Extracting locally..."):
            result = extract_document(pdf_path=pdf_path, fields=fields, model=model, provider="ollama")
    st.subheader("JSON")
    st.json(result, expanded=False)
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
