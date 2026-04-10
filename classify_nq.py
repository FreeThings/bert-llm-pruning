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
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
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
MAX_CHUNK_WORDS = 256          # words per chunk when recursing
CHUNK_OVERLAP_WORDS = 50       # sliding window overlap between chunks
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
    bert_score: Optional[float] = None
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return mean-pooled embeddings, shape (N, hidden_size)."""
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    def classify(self, query: str, text: str) -> str:
        """
        Ask the LLM whether `text` answers / is relevant to `query`.
        Returns 'yes' or 'no'.
        Uses tokenizer.apply_chat_template() so the prompt format is correct
        for any model (TinyLlama, Llama 3.2, etc.).
        """
        messages = [
            {"role": "system", "content": "You are a precise text classifier. Answer only with 'yes' or 'no'."},
            {"role": "user", "content": (
                f"Does the following passage answer or address this question?\n\n"
                f"Question: {query}\n\n"
                f"Passage: {text}\n\n"
                f"Answer with only 'yes' or 'no'."
            )},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
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

def split_into_chunks(
    text: str,
    max_words: int = MAX_CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split text into word-based chunks with a sliding window overlap."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max(1, max_words - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words])
        chunks.append(chunk)
        if i + max_words >= len(words):
            break
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
    question: str,
    bert_query_embedding: np.ndarray,
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

    question: the actual NQ question (e.g. "who played batman in the dark
        knight") — used for both LLM classification and BERT similarity.
    bert_query_embedding: pre-computed BERT embedding of `question`.
    """
    stats.total_nodes += 1
    text = node.text.strip()

    # --- Gating step ---
    # In "bert" mode, the parent may have already batch-scored this node.
    # If bert_score is still None we need to compute it (root node case).
    if mode == "bert":
        if node.bert_score is None:
            emb = embedder.embed([text])
            stats.bert_calls += 1
            node.bert_score = float(
                embedder.cosine_similarity(bert_query_embedding, emb)[0]
            )
        if node.bert_score < threshold:
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
        label = classifier.classify(question, text)
        stats.llm_calls += 1
        node.llm_label = label
        return label

    # --- Recurse: split into chunks ---
    chunks = split_into_chunks(text, max_words=MAX_CHUNK_WORDS)
    children = [TextNode(text=ct, depth=node.depth + 1) for ct in chunks]
    node.children = children

    # Batch BERT scoring for all children in one forward pass
    if mode == "bert":
        child_embs = embedder.embed([c.text for c in children])
        stats.bert_calls += len(children)
        sims = embedder.cosine_similarity(bert_query_embedding, child_embs)
        for child, sim in zip(children, sims):
            child.bert_score = float(sim)

    child_labels = []
    for child in children:
        label = classify_node_recursive(
            child, question, bert_query_embedding, embedder,
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
    Load N examples from Natural Questions (streams from HuggingFace).
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


def load_local_examples(path: str, n: int = 0) -> list[dict]:
    """
    Load pre-downloaded examples from a local JSON file.
    Use test_data/download_nq_sample.py to generate the file.
    """
    print(f"[NQ] Loading local data from {path} ...")
    with open(path) as f:
        examples = json.load(f)
    if n:
        examples = examples[:n]
    print(f"[NQ] Loaded {len(examples)} examples "
          f"({sum(e['has_answer'] for e in examples)} with answers)")
    return examples


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    examples: list[dict],
    question_embeddings: np.ndarray,
    embedder: BertEmbedder,
    classifier: LlamaClassifier,
    mode: str,
    threshold: float = SIMILARITY_THRESHOLD,
    prune_rate: float = 0.0,
) -> dict:
    """
    Run classification on all examples under the given mode.

    question_embeddings: shape (N, hidden_size), pre-computed BERT embeddings
        of the actual NQ questions (one per example).
    prune_rate: for 'random' mode, the fraction of nodes to prune. Should be
        set to the actual rate observed from a prior 'bert' run.

    Returns a results dict with predictions, ground truth, and stats.
    """
    print(f"\n{'='*60}")
    print(f" Mode: {mode.upper()}")
    if mode == "random":
        print(f" Prune rate: {prune_rate:.3f}")
    print(f"{'='*60}")

    stats = RunStats()
    t0 = time.time()

    predictions = []
    ground_truths = []

    for i, ex in enumerate(examples):
        # Use the actual question's BERT embedding for similarity scoring
        q_emb = question_embeddings[i : i + 1]  # shape (1, hidden_size)

        root = TextNode(text=ex["document_text"], depth=0)
        label = classify_node_recursive(
            root, ex["question"], q_emb, embedder,
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
        "--query", type=str, default=None,
        help="(Deprecated — ignored. The actual NQ question is now used "
             "automatically for both BERT gating and LLM classification.)",
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
        "--local-data", type=str, default=None,
        help="Path to a local JSON file of pre-downloaded NQ examples "
             "(see test_data/download_nq_sample.py)",
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
    if args.local_data:
        examples = load_local_examples(args.local_data, n=args.n)
    else:
        examples = load_nq_examples(
            n=args.n, split=args.split, max_doc_words=args.max_doc_words or None,
        )

    # Pre-compute question embeddings once (reused across all modes)
    print("[BERT] Pre-computing question embeddings ...")
    questions = [ex["question"] for ex in examples]
    # Batch in groups of 64 to avoid OOM on large N
    q_batches = [
        embedder.embed(questions[i : i + 64])
        for i in range(0, len(questions), 64)
    ]
    question_embeddings = np.concatenate(q_batches, axis=0)

    # Ensure bert runs before random so we can use the actual prune rate
    modes = list(args.modes)
    if "bert" in modes and "random" in modes:
        modes.sort(key=lambda m: 0 if m == "full" else (1 if m == "bert" else 2))

    # Run all requested modes and collect results
    all_results = []
    # Use actual bert prune rate when available; fall back to 1-threshold
    bert_prune_rate = 1.0 - args.threshold if "bert" not in modes else 0.0
    for mode in modes:
        result = run_evaluation(
            examples=examples,
            question_embeddings=question_embeddings,
            embedder=embedder,
            classifier=classifier,
            mode=mode,
            threshold=args.threshold,
            prune_rate=bert_prune_rate,
        )
        # Record actual prune rate from bert run for use by random
        if mode == "bert":
            s = result["stats"]
            bert_prune_rate = (
                s.pruned_nodes / s.total_nodes if s.total_nodes > 0 else 0.0
            )
            print(f"  [bert] Actual prune rate: {bert_prune_rate:.3f}")
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
