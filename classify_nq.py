"""
BERT-Augmented LLM Text Classification on Natural Questions
CS4964 Final Project — Greenbaum, McFarland, DeBoer

Algorithm:
  1. Load NQ dataset and a user-supplied query (classification prompt)
  2. Compute BERT embeddings for all candidate documents
  3. Score each document by cosine similarity to the query embedding
  4. Prune low-similarity documents
  5. Recursively split remaining docs into chunks, prune again
  6. Call Llama only on the surviving high-similarity chunks
  7. Aggregate chunk results to produce a per-document label

Evaluation compares three modes:
  - full    : LLM called on every document (no pruning)
  - bert    : BERT-guided pruning (our method)
  - random  : random pruning at the same rate as bert (control)
"""

import argparse
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer as BertAutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer as LlamaAutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_LLAMA_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TINYLLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BERT_MODEL = "bert-base-uncased"
SIMILARITY_THRESHOLD = 0.55   # cosine similarity gate for pruning
MAX_CHUNK_TOKENS = 256         # tokens per chunk when recursing
MIN_CHUNK_WORDS = 20           # stop recursing when chunk is this small
MAX_RECURSION_DEPTH = 3
MAX_DOC_WORDS = 500           # truncate documents for faster runs
NQ_TRAIN_SPLIT = "train"
NQ_VALIDATION_SPLIT = "validation"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TextNode:
    text: str
    depth: int = 0
    children: list = field(default_factory=list)
    llm_label: Optional[str] = None
    bert_score: float = 0.0
    was_pruned: bool = False


@dataclass
class RunStats:
    llm_calls: int = 0
    bert_calls: int = 0
    pruned_nodes: int = 0
    total_nodes: int = 0
    wall_time: float = 0.0


# ---------------------------------------------------------------------------
# BERT embeddings
# ---------------------------------------------------------------------------

class BertEmbedder:
    def __init__(self, model_name: str = BERT_MODEL, device: str = "cpu"):
        print(f"[BERT] Loading {model_name} ...")
        self.tokenizer = BertAutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return mean-pooled CLS embeddings, shape (N, hidden_size)."""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**enc)
        # mean pool over non-padding tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()
        embeddings = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return embeddings.cpu().numpy()

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a: (1, D), b: (N, D) → (N,)"""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return (b_norm @ a_norm.T).squeeze(-1)


# ---------------------------------------------------------------------------
# Llama LLM classifier
# ---------------------------------------------------------------------------

class LlamaClassifier:
    def __init__(self, model_name: str = DEFAULT_LLAMA_MODEL, device: str = "cpu"):
        print(f"[Llama] Loading {model_name} ...")

        # Use 4-bit quantization when a GPU is available to fit in VRAM
        if device.startswith("cuda"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                device_map="cpu",
            )

        tokenizer = LlamaAutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    def classify(self, query: str, text: str) -> str:
        """
        Ask the LLM whether `text` answers / is relevant to `query`.
        Returns 'yes' or 'no'.
        """
        prompt = (
            f"<|system|>\nYou are a precise text classifier. "
            f"Answer only with 'yes' or 'no'.\n</s>\n"
            f"<|user|>\nDoes the following passage answer or address this question?\n\n"
            f"Question: {query}\n\n"
            f"Passage: {text}\n\n"
            f"Answer with only 'yes' or 'no'.\n</s>\n"
            f"<|assistant|>\n"
        )
        result = self.pipe(prompt)[0]["generated_text"]
        # extract only the new tokens after the prompt
        answer = result[len(prompt):].strip().lower()
        if "yes" in answer:
            return "yes"
        return "no"


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def split_into_chunks(text: str, max_words: int = MAX_CHUNK_TOKENS) -> list[str]:
    """Split text into word-based chunks of at most max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
    return chunks


def extract_plain_text(document: dict) -> str:
    """
    Extract non-HTML tokens from an NQ document dict.
    The NQ `document` field has {'tokens': [{'token': str, 'is_html': bool}, ...]}.
    """
    tokens = document.get("tokens", [])
    words = [t["token"] for t in tokens if not t.get("is_html", False)]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Core recursive classification
# ---------------------------------------------------------------------------

def classify_node_recursive(
    node: TextNode,
    query: str,
    query_embedding: np.ndarray,
    embedder: BertEmbedder,
    classifier: LlamaClassifier,
    stats: RunStats,
    threshold: float,
    mode: str,          # "full" | "bert" | "random"
    prune_rate: float = 0.0,  # used only by "random" mode
) -> str:
    """
    Recursively classify a TextNode.
    Returns 'yes' or 'no'.
    """
    stats.total_nodes += 1
    text = node.text.strip()

    # --- Gating step ---
    if mode == "bert":
        emb = embedder.embed([text])
        stats.bert_calls += 1
        sim = float(embedder.cosine_similarity(query_embedding, emb)[0])
        node.bert_score = sim
        if sim < threshold:
            node.was_pruned = True
            stats.pruned_nodes += 1
            return "no"

    elif mode == "random":
        if random.random() < prune_rate:
            node.was_pruned = True
            stats.pruned_nodes += 1
            return "no"

    # --- Base case: small enough to call LLM ---
    words = text.split()
    if len(words) <= MIN_CHUNK_WORDS or node.depth >= MAX_RECURSION_DEPTH:
        label = classifier.classify(query, text)
        stats.llm_calls += 1
        node.llm_label = label
        return label

    # --- Recurse: split into chunks ---
    chunks = split_into_chunks(text, max_words=MAX_CHUNK_TOKENS)
    child_labels = []
    for chunk_text in chunks:
        child = TextNode(text=chunk_text, depth=node.depth + 1)
        node.children.append(child)
        label = classify_node_recursive(
            child, query, query_embedding, embedder,
            classifier, stats, threshold, mode, prune_rate,
        )
        child_labels.append(label)

    # A document is 'yes' if any surviving chunk is 'yes'
    result = "yes" if "yes" in child_labels else "no"
    node.llm_label = result
    return result


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_nq_examples(n: int = 100, split: str = NQ_VALIDATION_SPLIT, max_doc_words: int = MAX_DOC_WORDS) -> list[dict]:
    """
    Load N examples from Natural Questions.
    Each example: {'question': str, 'document_text': str, 'has_answer': bool}
    """
    print(f"[NQ] Loading Natural Questions ({split} split, first {n} examples) ...")
    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split=split,
        streaming=True,
    )

    examples = []
    for row in ds:
        question = row["question"]["text"]

        # Extract plain document text — tokens are parallel lists
        doc_tokens = row["document"]["tokens"]
        token_list = doc_tokens["token"]
        is_html_list = doc_tokens["is_html"]
        plain_words = [
            tok for tok, is_html in zip(token_list, is_html_list)
            if not is_html
        ]
        if max_doc_words:
            plain_words = plain_words[:max_doc_words]
        document_text = " ".join(plain_words)

        # Ground truth: does this document contain a long answer?
        # annotations fields are also parallel lists
        annotations = row["annotations"]
        long_answers = annotations["long_answer"]
        has_answer = any(
            la["start_token"] >= 0
            for la in long_answers
        )

        examples.append({
            "question": question,
            "document_text": document_text,
            "has_answer": has_answer,
        })

        if len(examples) >= n:
            break

    print(f"[NQ] Loaded {len(examples)} examples "
          f"({sum(e['has_answer'] for e in examples)} with answers)")
    return examples


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    examples: list[dict],
    query: str,
    embedder: BertEmbedder,
    classifier: LlamaClassifier,
    mode: str,
    threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    """
    Run classification on all examples under the given mode.
    Returns a results dict with predictions, ground truth, and stats.
    """
    print(f"\n{'='*60}")
    print(f" Mode: {mode.upper()}")
    print(f"{'='*60}")

    stats = RunStats()
    t0 = time.time()

    # Pre-compute query embedding once (used by all modes, but gating only in bert)
    query_embedding = embedder.embed([query])

    # Estimate prune_rate for random mode to match bert's pruning rate
    # We'll compute it after a dry bert run, or set it statically here
    prune_rate = 1.0 - threshold  # rough proxy

    predictions = []
    ground_truths = []

    for i, ex in enumerate(examples):
        root = TextNode(text=ex["document_text"], depth=0)
        label = classify_node_recursive(
            root, query, query_embedding, embedder,
            classifier, stats, threshold, mode, prune_rate,
        )
        predictions.append(label)
        ground_truths.append("yes" if ex["has_answer"] else "no")

        if (i + 1) % 10 == 0:
            print(f"  [{mode}] Processed {i+1}/{len(examples)} | "
                  f"LLM calls so far: {stats.llm_calls}")

    stats.wall_time = time.time() - t0

    report = classification_report(
        ground_truths, predictions, labels=["yes", "no"], output_dict=True
    )

    return {
        "mode": mode,
        "stats": stats,
        "predictions": predictions,
        "ground_truths": ground_truths,
        "classification_report": report,
    }


def print_results(results: dict):
    stats = results["stats"]
    report = results["classification_report"]
    print(f"\n--- Results: {results['mode'].upper()} ---")
    print(f"  LLM calls     : {stats.llm_calls}")
    print(f"  BERT calls    : {stats.bert_calls}")
    print(f"  Pruned nodes  : {stats.pruned_nodes}")
    print(f"  Total nodes   : {stats.total_nodes}")
    print(f"  Wall time (s) : {stats.wall_time:.1f}")
    print(f"  Accuracy      : {report['accuracy']:.3f}")
    print(f"  Precision(yes): {report['yes']['precision']:.3f}")
    print(f"  Recall(yes)   : {report['yes']['recall']:.3f}")
    print(f"  F1(yes)       : {report['yes']['f1-score']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BERT-Augmented LLM classification on Natural Questions"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Does this passage contain a direct answer to the question?",
        help="Classification query / prompt sent to the LLM",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of NQ examples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        choices=["train", "validation"],
        help="NQ dataset split to use",
    )
    parser.add_argument(
        "--threshold", type=float, default=SIMILARITY_THRESHOLD,
        help="BERT cosine similarity threshold for pruning (default: 0.55)",
    )
    parser.add_argument(
        "--llama-model", type=str, default=None,
        help="HuggingFace model ID for Llama (overrides --tiny)",
    )
    parser.add_argument(
        "--tiny", action="store_true",
        help="Use TinyLlama instead of Llama (no auth required)",
    )
    parser.add_argument(
        "--modes", nargs="+", default=["full", "bert", "random"],
        choices=["full", "bert", "random"],
        help="Which evaluation modes to run",
    )
    parser.add_argument(
        "--output", type=str, default="results.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--max-doc-words", type=int, default=MAX_DOC_WORDS,
        help="Truncate documents to this many words (default: 500, 0=no limit)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve model: --llama-model overrides --tiny, default is Llama
    if args.llama_model:
        llm_model = args.llama_model
    elif args.tiny:
        llm_model = TINYLLAMA_MODEL
    else:
        llm_model = DEFAULT_LLAMA_MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")
    print(f"[LLM]   Model: {llm_model}")

    # Load models
    embedder = BertEmbedder(device=device)
    classifier = LlamaClassifier(model_name=llm_model, device=device)

    # Load data
    examples = load_nq_examples(
        n=args.n, split=args.split, max_doc_words=args.max_doc_words or None,
    )

    # Run all requested modes and collect results
    all_results = []
    for mode in args.modes:
        result = run_evaluation(
            examples=examples,
            query=args.query,
            embedder=embedder,
            classifier=classifier,
            mode=mode,
            threshold=args.threshold,
        )
        print_results(result)
        all_results.append(result)

    # Summary table
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(f"{'Mode':<10} {'Accuracy':>10} {'F1(yes)':>10} {'LLM calls':>12} {'Wall(s)':>10}")
    print("-"*60)
    for r in all_results:
        s = r["stats"]
        rep = r["classification_report"]
        print(
            f"{r['mode']:<10} "
            f"{rep['accuracy']:>10.3f} "
            f"{rep['yes']['f1-score']:>10.3f} "
            f"{s.llm_calls:>12} "
            f"{s.wall_time:>10.1f}"
        )

    # Save JSON (convert stats dataclass to dict)
    output = []
    for r in all_results:
        s = r["stats"]
        output.append({
            "mode": r["mode"],
            "llm_calls": s.llm_calls,
            "bert_calls": s.bert_calls,
            "pruned_nodes": s.pruned_nodes,
            "total_nodes": s.total_nodes,
            "wall_time": s.wall_time,
            "accuracy": r["classification_report"]["accuracy"],
            "f1_yes": r["classification_report"]["yes"]["f1-score"],
            "precision_yes": r["classification_report"]["yes"]["precision"],
            "recall_yes": r["classification_report"]["yes"]["recall"],
        })
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Output] Results saved to {args.output}")


if __name__ == "__main__":
    main()
