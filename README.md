# Local Invoice Intelligence 🧾

An enterprise-grade, privacy-first invoice extraction pipeline built entirely on local Small Language Models (SLMs). 

This project demonstrates a production-ready architecture that decouples **visual perception** from **semantic reasoning**, achieving cloud-level accuracy (77%+) while running 100% locally on Apple Silicon (M-series) hardware without VRAM swap thrashing or GGML tensor crashes.

## Executive Summary

Extracting structured JSON from invoices is a notoriously difficult task due to chaotic spatial layouts and floating text blocks. Relying on heavy Vision-Language Models (VLMs) for this task locally often results in hallucinated schema structures, compounding errors, and severe hardware bottlenecks.

This project solves this by implementing a **Hybrid Perception & Agentic Reasoning Architecture**:
1. **The "Eyes" (Deterministic Perception):** Uses `pdfplumber` to extract a 2D spatial text grid (preserving physical layout via whitespace) with a Tesseract OCR fallback for flat scans.
2. **The "Brain" (LLM Auditor):** Uses strictly constrained local SLMs (`llama3.1:8b` and `llama3.2:3b`) equipped with a cognitive scratchpad to perform mathematical auditing before mapping to a rigid Pydantic JSON schema.

## Architecture & Engineering Breakthroughs

### 1. Decoupling Perception from Analysis
Initial experiments utilizing VLMs (like GLM-OCR and Qwen-VL) to directly output JSON resulted in high latency and context pollution across multi-page documents. 

By forcing the pipeline to extract raw, spatially-aware text first, the LLM is freed from visual processing overhead. It operates entirely on high-density semantic text, drastically reducing context window pollution and completely eliminating the Apple Silicon Metal backend crashes associated with heavy vision tensors.

### 2. Solving the Spatial Hallucination Problem
Standard document parsers (like Docling or MinerU) excel at dense academic PDFs but fail on invoices because they force floating text blocks into rigid Markdown tables (`|---|---|`), severing the semantic relationship between a vendor name and its adjacent address.

**The Fix:** This pipeline uses `pdfplumber` with `layout=True` to pad the text with exact whitespace. This preserves the visual columns in plaintext, allowing the LLM to successfully differentiate a line-item subtotal from a final gross total located on the opposite side of the page.

### 3. The "Scratchpad" Grammar Constraint
LLMs inherently struggle with zero-shot rigid JSON generation when math is involved. To prevent the model from blindly extracting subtotals, the Pydantic schema forces the model to generate a `reasoning_process` string *first*. The LLM uses this scratchpad to explicitly verify dates and double-check mathematical totals before it is allowed to populate the final JSON keys. 

## Evaluation & Benchmarks

The system was evaluated using a custom semantic matching script (resilient to OCR noise, currency symbols, and date formatting artifacts) across a complex **297-document benchmark** from the DocILE dataset. 

Two distinct architectures were tested to establish the ultimate Accuracy vs. Latency frontier on an M5 Mac with 24GB Unified Memory.

### Option A: The "Accuracy" Backend (Asynchronous Processing)
Optimized for nightly batch processing where precision is paramount.
* **Model:** `llama3.1:8b`
* **Average Latency:** ~15 seconds / document

| Extraction Field | Accuracy |
| :--- | :--- |
| **Vendor Name** | 75.58% |
| **Vendor Address** | 78.20% |
| **Total Gross Amount** | 70.03% |
| **Date Issued** | 84.18% |
| **Total Average** | **77.00%** |

### Option B: The "Speed" Edge (Real-Time Synchronous)
Optimized for high-throughput, user-facing applications requiring immediate preliminary extraction.
* **Model:** `llama3.2:3b`
* **Average Latency:** ~5 seconds / document

| Extraction Field | Accuracy |
| :--- | :--- |
| **Vendor Name** | 71.13% |
| **Vendor Address** | 75.12% |
| **Total Gross Amount** | 67.68% |
| **Date Issued** | 78.11% |
| **Total Average** | **73.01%** |

## 🛠️ Tech Stack
* **Orchestration:** Python, Concurrent Futures (Multithreading with Thermal Cooldowns)
* **Local Inference:** Ollama 
* **Schema Validation:** Pydantic
* **Spatial Parsing:** `pdfplumber`, `PyMuPDF` (fitz)
* **OCR Fallback:** Tesseract (`pytesseract`)

## Future Roadmap
1. **Microservice Deployment:** Wrap the extraction engine in a `FastAPI` endpoint for seamless integration into modern web stacks.
2. **Dynamic Vision Routing:** Implement a programmatic Validation Gate that checks the LLM's JSON output. If a critical field is missed, dynamically route *only* that specific page to a lightweight VLM (e.g., `llama3.2-vision:11b`) for surgical error recovery, pushing total accuracy past 80% without sacrificing baseline latency.

---
*Architected and developed by Sanjay.*