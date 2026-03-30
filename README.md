# BERT-Augmented LLM Text Classification Tasks: Prune or Be Pruned

CS4964 Final Project — Josh Greenbaum, Alex McFarland, Aiden DeBoer

## Overview

This project explores whether BERT embeddings can serve as a cheap gating mechanism to reduce the number of expensive LLM calls needed for large-scale text classification. We use cosine similarity between BERT embeddings to prune irrelevant text nodes before they reach the LLM, recursively narrowing down documents into smaller chunks and only classifying the most promising ones.

The approach is evaluated on the [Google Natural Questions](https://github.com/google-research-datasets/natural-questions) dataset using Meta's Llama as the LLM classifier.

## Algorithm

1. **Compute BERT embeddings** for each candidate document
2. **Compute similarity** between each document embedding and the query prompt embedding
3. **Prune** documents with low similarity
4. **Split** remaining documents into chunks and compute BERT embeddings for each chunk
5. **Prune again** — repeat steps 4–5 recursively until chunks are sentence/paragraph sized
6. **Call the LLM** only on surviving chunks; label each document based on chunk results

Estimated relevance score: `es = LLM_score × BERT_similarity`

## Setup

### Prerequisites

- Python 3.10+
- A [HuggingFace](https://huggingface.co/) account with access to the Llama model

### Install dependencies

```bash
pip install -r requirements.txt
```

### Authenticate with HuggingFace

You need to accept the [Llama license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on HuggingFace, then log in:

```bash
huggingface-cli login
```

## Usage

### Basic run (50 examples, all 3 modes)

```bash
python classify_nq.py --n 50
```

### Custom threshold and modes

```bash
python classify_nq.py --n 100 --threshold 0.60 --modes bert random
```

### Specify a different Llama model

```bash
python classify_nq.py --llama-model meta-llama/Llama-3.2-1B-Instruct --n 20
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | `"Does this passage contain a direct answer to the question?"` | Classification prompt sent to the LLM |
| `--n` | `50` | Number of NQ examples to evaluate |
| `--split` | `validation` | NQ dataset split (`train` or `validation`) |
| `--threshold` | `0.55` | BERT cosine similarity cutoff for pruning |
| `--llama-model` | `meta-llama/Llama-3.2-1B-Instruct` | HuggingFace model ID |
| `--modes` | `full bert random` | Evaluation modes to run |
| `--output` | `results.json` | Path for JSON results |
| `--seed` | `42` | Random seed |

## Evaluation

The script compares three configurations:

| Mode | Description |
|------|-------------|
| **full** | LLM called on every document node (no pruning) — baseline |
| **bert** | BERT-guided pruning (our method) — only LLM-classify nodes above the similarity threshold |
| **random** | Random pruning at a comparable rate — control |

Metrics reported: accuracy, precision, recall, F1, total LLM calls, BERT calls, pruned node count, and wall-clock time.

Results are printed as a summary table and saved to `results.json`.

## Notes

- On GPU the script uses 4-bit quantization (via `bitsandbytes`) to fit Llama in VRAM.
- On CPU it runs in fp32 — use a small model (e.g., `Llama-3.2-1B-Instruct`) and a low `--n` for testing.
- The Natural Questions dataset streams from HuggingFace so no manual download is needed.
