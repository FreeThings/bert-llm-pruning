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
6. **Call the LLM** only on surviving chunks in a single batched call; label each document based on chunk results

Threshold tightens at each depth level: `τ(depth) = base + depth × step`

## Setup

### Prerequisites

- Python 3.10+
- A [HuggingFace](https://huggingface.co/) account (only required for Llama; TinyLlama is open)
- **A CUDA-capable NVIDIA GPU is strongly recommended** (see below)

### Install dependencies

```bash
pip install -r requirements.txt
```

### GPU setup (strongly recommended)

The default `pip install torch` gives you a CPU-only build. On CPU, a 50-example run takes 2–3 hours per mode. On a modern GPU it takes under 20 minutes total.

Check whether you already have GPU support:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this prints `False`, reinstall PyTorch with CUDA support. Use the CUDA 12.6 build — it works with any driver that supports CUDA 12.x or higher (driver ≥ 525):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
```

Then verify:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
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

## Time estimates

| Setup | n | Modes | Estimated time |
|-------|---|-------|---------------|
| CPU, TinyLlama | 10 | bert | ~10–20 min |
| CPU, TinyLlama | 50 | full + bert + random | ~6–9 hours |
| GPU (8GB), TinyLlama | 50 | full + bert + random | ~15–25 min |
| GPU (8GB), Llama 3.2 1B | 50 | full + bert + random | ~20–35 min |
| GPU (8GB), TinyLlama | 50 | full + bert + random + tune | ~45–75 min |

LLM inference is the bottleneck — GPU makes it 10–20× faster. BERT scoring is fast on either device.

## Usage

### Quick start with TinyLlama (no auth required)

```bash
python classify_nq.py --tiny --n 20
```

### Run with Llama (requires HuggingFace auth)

```bash
python classify_nq.py --n 50
```

### With threshold tuning (recommended for best results)

Runs a coarse + fine grid search to find optimal pruning thresholds before evaluation:

```bash
python classify_nq.py --tiny --n 150 --tune --output results_tuned.json
```

### Run bert-only with custom threshold

```bash
python classify_nq.py --tiny --n 100 --threshold 0.25 --modes bert
```

### Smoke test (verify setup, no meaningful metrics)

```bash
python classify_nq.py --tiny --n 5 --max-doc-words 200
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--n` | `50` | Number of NQ examples to evaluate (use ≥150 for reliable tuning) |
| `--local-data` | — | Path to local JSON of pre-downloaded NQ examples (skips streaming) |
| `--save-data` | — | Save loaded examples to a JSON file for offline reuse |
| `--split` | `validation` | NQ dataset split (`train` or `validation`) |
| `--threshold` | `0.23` | Base BERT cosine similarity cutoff for pruning (tuned) |
| `--threshold-step` | `0.08` | How much the threshold tightens per recursion depth (tuned) |
| `--aggregation` | `hard` | `hard` = any yes → yes (default); `soft` = weighted mean of BERT scores |
| `--doc-score-threshold` | `0.0` | Soft aggregation cutoff (unused in hard mode) |
| `--tiny` | off | Use TinyLlama instead of Llama (no auth required) |
| `--llama-model` | `meta-llama/Llama-3.2-1B-Instruct` | HuggingFace model ID (overrides `--tiny`) |
| `--max-doc-words` | `0` | Truncate documents to N words (0 = no limit; use 200 for quick smoke tests) |
| `--modes` | `full bert random` | Evaluation modes to run |
| `--output` | `results.json` | Path for JSON results |
| `--tune` | off | Run coarse + fine grid search to find best thresholds before evaluation |
| `--train-split` | `0.8` | Fraction of data used for training during `--tune` |
| `--seed` | `42` | Random seed |

## Evaluation

The script compares three configurations:

| Mode | Description |
|------|-------------|
| **full** | LLM called on every document node (no pruning) — baseline |
| **bert** | BERT-guided pruning (our method) — only LLM-classify nodes above the similarity threshold |
| **random** | Random pruning at the same rate as bert — control |

Metrics reported: accuracy, precision, recall, F1, total LLM calls, BERT calls, pruned node count, and wall-clock time.

Results are printed as a summary table and saved to `results.json`.

## Local test data

Streaming NQ from HuggingFace can be slow on first load. Save examples locally once and reuse them:

```bash
# Stream and save 200 examples (one-time)
python classify_nq.py --tiny --n 200 --save-data nq_200.json --modes bert --output /dev/null

# All subsequent runs load instantly
python classify_nq.py --tiny --local-data nq_200.json --modes bert
```

Alternatively, use the standalone download script in `test_data/`:

```bash
python test_data/download_nq_sample.py --n 50
python classify_nq.py --tiny --local-data test_data/nq_sample_50.json --modes bert
```

## Notes

- **TinyLlama vs Llama**: Use `--tiny` for quick testing without auth. For final results, use Llama (the default) which requires HuggingFace authentication and license acceptance.
- **BERT model**: Uses `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity scoring — significantly better cosine scores than a generic BERT model for question-passage matching.
- On GPU the script uses 4-bit quantization (via `bitsandbytes`) to fit the LLM in VRAM.
- On CPU it runs in fp32 — use `--tiny` and `--max-doc-words 200` for faster iteration.
- The Natural Questions dataset streams from HuggingFace so no manual download is needed, but `--local-data` is recommended for faster iteration (see above).
