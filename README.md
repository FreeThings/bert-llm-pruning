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
- A [HuggingFace](https://huggingface.co/) account

### Install dependencies

```bash
pip install -r requirements.txt
```

### Authenticate with HuggingFace (required for Llama)

If using Llama (the default), you need to:

1. Create a HuggingFace account at https://huggingface.co/
2. Go to https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct and accept the license
3. Create an access token at https://huggingface.co/settings/tokens
4. Log in from your terminal:

```bash
pip install huggingface_hub
huggingface-cli login
```

If you're using TinyLlama (`--tiny` flag), no authentication is needed.

## Usage

### Quick start with TinyLlama (no auth required)

```bash
python classify_nq.py --tiny --n 20
```

### Run with Llama (requires HuggingFace auth)

```bash
python classify_nq.py --n 50
```

### Custom threshold and modes

```bash
python classify_nq.py --tiny --n 100 --threshold 0.60 --modes bert random
```

### Faster CPU testing (truncate long documents)

```bash
python classify_nq.py --tiny --n 10 --max-doc-words 200
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | `50` | Number of NQ examples to evaluate |
| `--local-data` | — | Path to local JSON of pre-downloaded NQ examples (skips streaming) |
| `--split` | `validation` | NQ dataset split (`train` or `validation`) |
| `--threshold` | `0.55` | BERT cosine similarity cutoff for pruning |
| `--tiny` | off | Use TinyLlama instead of Llama (no auth required) |
| `--llama-model` | `meta-llama/Llama-3.2-1B-Instruct` | HuggingFace model ID (overrides `--tiny`) |
| `--max-doc-words` | `500` | Truncate documents to N words (0 = no limit) |
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

## Local test data

Streaming NQ from HuggingFace can be slow. To avoid this, download a sample once and reuse it:

```bash
# Download 50 examples to test_data/nq_sample_50.json (one-time, takes a few minutes)
python test_data/download_nq_sample.py --n 50

# Run using the local data (instant loading)
python classify_nq.py --tiny --local-data test_data/nq_sample_50.json --modes bert random
```

The download script accepts `--n`, `--split`, `--max-doc-words`, and `--seed` flags.

## Notes

- **TinyLlama vs Llama**: Use `--tiny` for quick testing without auth. For final results, use Llama (the default) which requires HuggingFace authentication and license acceptance.
- On GPU the script uses 4-bit quantization (via `bitsandbytes`) to fit Llama in VRAM.
- On CPU it runs in fp32 — use `--tiny` and `--max-doc-words 200` for faster iteration.
- The Natural Questions dataset streams from HuggingFace so no manual download is needed, but `--local-data` is recommended for faster iteration (see above).
