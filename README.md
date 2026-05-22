# Local Invoice Intelligence 📄🔍

A privacy-first, zero-cost, local extraction pipeline for semi-structured business documents such as invoices, receipts, and purchase orders.

This project explores moving away from expensive cloud-based Document AI APIs like [Azure Document Intelligence](https://azure.microsoft.com/en-in/products/ai-services/ai-document-intelligence?utm_source=chatgpt.com) or [AWS Textract](https://aws.amazon.com/textract/?utm_source=chatgpt.com) by leveraging local Vision Language Models (VLMs) to process sensitive financial data completely offline.

---

## The Philosophy

Data extraction is rarely just a model problem. It is a **systems problem**.

While a single-pass extraction from an LLM/VLM provides a baseline, achieving production-grade reliability requires:

* Layout parsing
* Multi-step extraction
* Validation loops
* Semantic verification
* Error recovery pipelines

This repository serves as the foundational baseline and evaluation framework for building a highly accurate, multi-model, agentic local extraction system.

---

## Current Baseline Benchmark (Zero-Shot)

The initial experiment runs a zero-shot extraction on a 50-document subset of the [DocILE Dataset](https://github.com/rossumai/docile/tree/main) while enforcing strict JSON schema outputs.

### Baseline Configuration

* **Model:** Llama 3.2 Vision (11B) via Ollama
* **Compute:** Unified Memory Architecture (Apple Silicon M-Series)
* **Target Fields:**

  * Vendor Name
  * Vendor Address
  * Total Gross Amount
  * Issue Date
* **Evaluation Method:**

  * Fuzzy token-set matching for text
  * Mathematical float comparison for amounts
  * Datetime parsing for dates

### Initial Results

```text
====== TRUE EXTRACTION ACCURACY ======

vendor_name           : 68.08%
vendor_address        : 70.14%
amount_total_gross    : 60.00%
date_issue            : 60.00%

======================================
Total average: 64.55%
```

> Note: This is an unoptimized, single-pass baseline intended to establish a benchmark for future systemic improvements.

---

## Tech Stack

### Inference Engine

* Ollama (local runtime)

### Dependency Management

* uv (lightning-fast Python package manager)

### Structured Output

* Pydantic (strict schema enforcement)

### Document Processing

* PyMuPDF (`fitz`)
* Resolution clamping to prevent VRAM context-window overflows

### Evaluation

* RapidFuzz
* python-dateutil
* Semantic evaluation pipelines

---

## Repository Structure

```plaintext
local_invoice_intelligence/
├── data/
│   └── eval_subset_ids_sorted.json   # Pre-sorted target documents (by area)
│
├── results/
│   └── eval_report.json              # VLM JSON extraction outputs
│
├── src/
│   ├── config.py                     # Project paths and VLM toggles
│   ├── schema.py                     # Pydantic schemas for JSON enforcement
│   ├── sort_dataset.py               # Curriculum batching utility
│   ├── eval_runner.py                # Multithreaded local inference pipeline
│   └── evaluate_metrics.py           # Semantic scoring logic
│
├── pyproject.toml                    # uv dependencies
└── README.md
```

---

## Quickstart & Setup

### 1. Install Prerequisites

Download and install [Ollama](https://ollama.com/?utm_source=chatgpt.com).

Pull the baseline vision model:

```bash
ollama run llama3.2-vision
```

Install `uv` if you have not already.

---

### 2. Clone and Setup Environment

```bash
git clone https://github.com/YOUR_USERNAME/local_invoice_intelligence.git

cd local_invoice_intelligence

uv sync

source .venv/bin/activate
```

---

### 3. Dataset Preparation

This project requires the [DocILE Dataset](https://github.com/rossumai/docile/tree/main).

Download the dataset and update the `DOCILE_DATASET_PATH` in `src/config.py` to point to your local unzipped dataset directory.

---

### 4. Run the Pipeline

#### Optional: Sort the dataset by physical document dimensions

This helps optimize the local VRAM context-window queue.

```bash
uv run python src/sort_dataset.py
```

#### Run the inference pipeline

Includes a multithreaded watchdog system to handle VLM stalling.

```bash
uv run python src/eval_runner.py
```

#### Evaluate results semantically

```bash
uv run python src/evaluate_metrics.py
```

---

## Roadmap

* [x] Establish baseline zero-shot local VLM architecture
* [x] Implement semantic evaluation (fuzzy matching, datetime parsing)
* [x] Handle architectural edge cases

  * VRAM context-window clamping
  * Thread timeouts
* [ ] Implement multi-step reasoning and prompt scratchpads
* [ ] Explore agentic orchestration (e.g. LangGraph) with auditor nodes to validate mathematical totals against extracted line items
* [ ] Build a lightweight local UI for drag-and-drop batch document processing
