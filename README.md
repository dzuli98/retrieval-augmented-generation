# Multi-Agent RAG System

A minimal prototype of a conflict-aware RAG system using the RAMDocs dataset.

## Overview

This system handles conflicting information in retrieved documents by:
1. **Retrieving** relevant documents using FAISS vector search
2. **Analyzing** each document to extract claims (Document Analyzer Agent)  
3. **Detecting conflicts** and reconciling them (Mediator Agent)
4. **Generating** final answer with explanation and evidence

## Project Structure

```
ai-agent/
├── docs/
│   └── ARCHITECTURE.md     # Design documentation
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration
│   ├── models.py           # Data models (Document, Claim, FinalAnswer)
│   ├── data_loader.py      # RAMDocs loading + FAISS indexing
│   ├── pipeline.py         # Main pipeline orchestration
│   ├── logger.py           # Logging utility
│   ├── evaluate.py         # Evaluation against ground truth
│   └── agents/
│       ├── __init__.py
│       ├── analyzer.py     # Document analysis agent (parallel)
│       └── mediator.py     # Conflict detection & resolution
├── main.py                 # Entry point
├── pyproject.toml          # uv dependency manager config
├── requirements.txt
└── README.md
```

## Installation

```bash
 # Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run demo (loads RAMDocs, runs first question)
python -m main --demo

# Evaluate against RAMDocs ground truth
python -m main --evaluate --samples 10

# Single query
python -m main --query "What is the capital of France?"
```

Set your API key by creating a `.env` file in the project root with `OPENAI_API_KEY=your_key_here`. 
See `.env.example` for the required format. Alternatively, set the `OPENAI_API_KEY` environment variable or pass `--api-key` directly.

## Example Output

```
============================================================
QUERY: What is the capital of Australia?
============================================================

📝 ANSWER: Canberra
   Confidence: 90%

📚 SUPPORTING EVIDENCE:
   • [q0_d1] confidence=95%, reliability=100% (correct)
     "Canberra is the capital city of Australia..."
   • [q0_d3] confidence=85%, reliability=70% (unknown)
     "The capital of Australia is Canberra..."

❌ REJECTED CLAIMS:
   • [q0_d2] "Sydney" - Conflict with majority (source=noise)

🔍 RECONCILIATION:
   Resolved by majority vote. 2 sources support 'Canberra', 1 rejected.

📋 EXECUTION TRACE:
   → [10:30:15] Step 1: Retrieving documents for: 'What is the capital...'
   → [10:30:16] Step 2: Analyzing documents to extract claims...
   → [10:30:18] Step 3: Detecting conflicts and reconciling...
   → [10:30:18] ✅ Final answer: 'Canberra' (confidence=90%)
```

## Conflict Resolution

Conflicts are resolved using **majority vote** weighted by `confidence × reliability`.
This approach avoids extra LLM calls while effectively downweighting unreliable sources.

### Reliability Scoring

Claims are weighted by source reliability based on RAMDocs labels:

| Label | Reliability | Description |
|-------|-------------|-------------|
| `correct` | 100% | Fully trusted |
| `unknown` | 70% | Neutral |
| `noise` | 30% | Downweighted |
| `misinfo` | 10% | Heavily penalized |

## Requirements Met

- ✅ **Indexing/retrieval**: Ingests RAMDocs subset, FAISS vector search
- ✅ **Per-document analysis**: Parallel extraction with confidence scores
- ✅ **Mediator agent**: Conflict detection with reliability-weighted resolution
- ✅ **Metadata handling**: Source types (correct/misinfo/noise) used in scoring
- ✅ **Answer generation**: Human-readable output with evidence and explanation
- ✅ **Traceability**: Full execution trace + Python logging
- ✅ **Evaluation**: Standard QA metrics (Accuracy, EM, F1, Misinformation Rate)

## Evaluation Metrics

The evaluation uses standard QA metrics following SQuAD evaluation:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction matching any gold answer (fuzzy, F1 ≥ 0.8) |
| **Exact Match (EM)** | Strict string equality after normalization |
| **F1 Score** | Token-level overlap with best gold answer |
| **Misinformation Rate** | Fraction matching known wrong answers (lower is better!) |
