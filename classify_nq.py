"""
BERT-Augmented LLM Text Classification on Natural Questions
CS4964 Final Project — Greenbaum, McFarland, DeBoer

Algorithm (gate-then-chunk with depth-scaled thresholds):
  1. Load NQ dataset
  2. Split each document into BERT-max windows (~380 words)
     (BERT can only see 512 tokens, so scoring full docs is unreliable)
  3. BERT-score each window against the query, prune below threshold
  4. Split survivors into paragraphs, score and prune (tighter threshold)
  5. Surviving leaf chunks sent to Llama
  6. Aggregate: soft (weight by BERT score) or hard (any yes = yes)
  Threshold scales with depth: τ(d) = base + d * step

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
SIMILARITY_THRESHOLD = 0.40   # base cosine similarity gate for pruning
THRESHOLD_STEP = 0.10         # increase threshold by this much per depth level
BERT_MAX_WORDS = 380           # ~512 tokens — max BERT can actually see
MIN_CHUNK_WORDS = 40           # stop recursing — "few sentences" leaf size
MAX_RECURSION_DEPTH = 3
MAX_DOC_WORDS = 0             # 0 = no truncation (use full document length)
DOC_SCORE_THRESHOLD = 0.35    # soft aggregation: min bert_score for a "yes" chunk to count

# Chunk sizes per depth level — defines the funnel shape
# depth 0 (doc):               split into BERT-max windows (~380 words)
# depth 1 (BERT-max windows):  gate, then split survivors into paragraphs
# depth 2 (paragraphs):        gate, then leaf → hits LLM
CHUNK_WORDS_BY_DEPTH = [BERT_MAX_WORDS, 150, 40]
CHUNK_OVERLAP_WORDS = 20       # sliding window overlap between chunks
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

def get_chunk_size_for_depth(depth: int) -> int:
    """Return the target chunk size (in words) for children at the given depth."""
    if depth < len(CHUNK_WORDS_BY_DEPTH):
        return CHUNK_WORDS_BY_DEPTH[depth]
    return MIN_CHUNK_WORDS


def split_into_chunks(
    text: str,
    max_words: int,
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
    base_threshold: float,
    threshold_step: float,
    mode: str,          # "full" | "bert" | "random"
    prune_rate: float = 0.0,  # used only by "random" mode
    aggregation: str = "soft",  # "soft" or "hard"
    doc_score_threshold: float = DOC_SCORE_THRESHOLD,
    gates_passed: int = 0,  # how many BERT gates ancestors have passed through
) -> tuple[str, float]:
    """
    Recursive classifier with depth-scaled BERT gating.

    Gate-then-chunk: at every level, BERT scores the current node first.
    If it fails the threshold, the entire subtree is pruned — no chunking,
    no children, no LLM calls.  This is where the cost savings come from:
    killing whole branches early.

    The threshold tightens at each depth level:
        threshold(depth) = base_threshold + depth * threshold_step

    Depth 0 (document):          too large for BERT — skip gate, just chunk
    Depth 1 (BERT-max windows):  τ = base — first real gate, prune whole branches
    Depth 2 (paragraphs):        τ = base + step — tighter
    Depth 3+ (sentences):        τ = base + 2*step — strictest, leaf → LLM

    Returns (label, estimated_score):
      - label: 'yes' or 'no'
      - estimated_score: best (bert_score * llm_yes) across surviving chunks
    """
    stats.total_nodes += 1
    text = node.text.strip()
    words = text.split()

    # Skip the BERT gate if the text exceeds BERT's window — the embedding
    # would only see the first ~380 words, making the score unreliable.
    # Just chunk it down to BERT-max size and gate the children instead.
    is_oversized = len(words) > BERT_MAX_WORDS
    depth_threshold = base_threshold + gates_passed * threshold_step

    # --- Gate: BERT scores this node BEFORE we chunk or call LLM ---
    if not is_oversized:
        if mode == "bert":
            if node.bert_score is None:
                emb = embedder.embed([text])
                stats.bert_calls += 1
                node.bert_score = float(
                    embedder.cosine_similarity(bert_query_embedding, emb)[0]
                )
            if node.bert_score < depth_threshold:
                node.was_pruned = True
                stats.pruned_nodes += 1
                return "no", 0.0
        elif mode == "random":
            if random.random() < prune_rate:
                node.was_pruned = True
                stats.pruned_nodes += 1
                return "no", 0.0
        gates_passed += 1

    # --- Base case: small enough to send to LLM ---
    if len(words) <= MIN_CHUNK_WORDS or node.depth >= MAX_RECURSION_DEPTH:
        label = classifier.classify(question, text)
        stats.llm_calls += 1
        node.llm_label = label
        score = (node.bert_score or 0.0) if label == "yes" else 0.0
        return label, score

    # --- Recurse: split into depth-appropriate chunks ---
    child_chunk_size = get_chunk_size_for_depth(node.depth)
    chunks = split_into_chunks(text, max_words=child_chunk_size)

    # If splitting produced a single chunk (text already smaller than chunk
    # size), don't create an identical child — that would waste a BERT call
    # and apply a stricter threshold to unchanged text.  Treat as a leaf.
    if len(chunks) == 1:
        label = classifier.classify(question, text)
        stats.llm_calls += 1
        node.llm_label = label
        score = (node.bert_score or 0.0) if label == "yes" else 0.0
        return label, score

    children = [TextNode(text=ct, depth=node.depth + 1) for ct in chunks]
    node.children = children

    # Batch BERT scoring for children (batch in groups of 32 to avoid OOM)
    if mode == "bert":
        child_texts = [c.text for c in children]
        emb_batches = [
            embedder.embed(child_texts[i : i + 32])
            for i in range(0, len(child_texts), 32)
        ]
        child_embs = np.concatenate(emb_batches, axis=0)
        stats.bert_calls += len(children)
        sims = embedder.cosine_similarity(bert_query_embedding, child_embs)
        for child, sim in zip(children, sims):
            child.bert_score = float(sim)

    child_scores = []
    child_labels = []
    for child in children:
        label, score = classify_node_recursive(
            child, question, bert_query_embedding, embedder,
            classifier, stats, base_threshold, threshold_step,
            mode, prune_rate, aggregation, doc_score_threshold, gates_passed,
        )
        child_labels.append(label)
        child_scores.append(score)

    # --- Aggregation ---
    best_score = max(child_scores) if child_scores else 0.0

    if aggregation == "soft":
        result = "yes" if best_score > doc_score_threshold else "no"
    else:
        result = "yes" if "yes" in child_labels else "no"

    node.llm_label = result
    return result, best_score


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
    threshold_step: float = THRESHOLD_STEP,
    prune_rate: float = 0.0,
    aggregation: str = "soft",
    doc_score_threshold: float = DOC_SCORE_THRESHOLD,
) -> dict:
    """
    Run classification on all examples under the given mode.

    question_embeddings: shape (N, hidden_size), pre-computed BERT embeddings
        of the actual NQ questions (one per example).
    prune_rate: for 'random' mode, the fraction of nodes to prune. Should be
        set to the actual rate observed from a prior 'bert' run.
    threshold_step: how much to increase threshold per recursion depth.
    aggregation: 'soft' (weight by bert score) or 'hard' (any yes = yes).

    Returns a results dict with predictions, ground truth, and stats.
    """
    print(f"\n{'='*60}")
    print(f" Mode: {mode.upper()}")
    if mode == "bert":
        print(f" Base threshold: {threshold:.3f}, step: {threshold_step:.3f}")
        print(f" Aggregation: {aggregation}")
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
        label, _score = classify_node_recursive(
            root, ex["question"], q_emb, embedder,
            classifier, stats, threshold, threshold_step,
            mode, prune_rate, aggregation, doc_score_threshold,
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


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def serialize_tree(node: TextNode) -> dict:
    """Serialize a scored tree into a dict for threshold replay."""
    return {
        "bert_score": node.bert_score,
        "llm_label": node.llm_label,
        "depth": node.depth,
        "is_leaf": len(node.children) == 0,
        "children": [serialize_tree(c) for c in node.children],
    }


def replay_tree(node_data: dict, base_threshold: float, threshold_step: float,
                doc_score_threshold: float, gates_passed: int = 0) -> tuple[str, float, int]:
    """
    Simulate pruning on a pre-scored tree node. Returns (label, score, llm_calls).
    No model calls — just replays the threshold logic on cached scores.
    """
    bert_score = node_data["bert_score"]
    depth_threshold = base_threshold + gates_passed * threshold_step

    # Gate: only if BERT could score this node (not oversized)
    if bert_score is not None:
        if bert_score < depth_threshold:
            return "no", 0.0, 0
        gates_passed += 1

    # Leaf
    if node_data["is_leaf"]:
        label = node_data["llm_label"]
        score = bert_score if (label == "yes" and bert_score is not None) else 0.0
        return label, score, 1

    # Recurse into children
    child_scores = []
    total_llm = 0
    for child in node_data["children"]:
        label, score, llm = replay_tree(
            child, base_threshold, threshold_step, doc_score_threshold, gates_passed
        )
        child_scores.append(score)
        total_llm += llm

    best_score = max(child_scores) if child_scores else 0.0
    result = "yes" if best_score > doc_score_threshold else "no"
    return result, best_score, total_llm


def run_tuning(
    examples: list[dict],
    question_embeddings: np.ndarray,
    embedder: BertEmbedder,
    classifier: LlamaClassifier,
    train_split: float = 0.8,
    seed: int = 42,
) -> dict:
    """
    Tune thresholds using train/test split.

    1. Split examples into train/test
    2. Run full tree (no pruning) on train set to collect BERT scores + LLM labels
    3. Sweep threshold combos on train trees (free — no model calls)
    4. Pick best params by F1
    5. Evaluate on test set with best params

    Returns dict with best params and test results.
    """
    rng = random.Random(seed)

    # --- Train/test split (stratified by has_answer) ---
    pos = [i for i, e in enumerate(examples) if e["has_answer"]]
    neg = [i for i, e in enumerate(examples) if not e["has_answer"]]
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_train_pos = max(1, int(len(pos) * train_split))
    n_train_neg = max(1, int(len(neg) * train_split))

    train_idx = set(pos[:n_train_pos] + neg[:n_train_neg])
    test_idx = set(pos[n_train_pos:] + neg[n_train_neg:])

    train_examples = [examples[i] for i in sorted(train_idx)]
    test_examples = [examples[i] for i in sorted(test_idx)]
    train_q_embs = np.array([question_embeddings[i] for i in sorted(train_idx)])
    test_q_embs = np.array([question_embeddings[i] for i in sorted(test_idx)])

    print(f"\n{'='*60}")
    print(f" THRESHOLD TUNING")
    print(f"{'='*60}")
    print(f"  Train: {len(train_examples)} examples "
          f"({sum(e['has_answer'] for e in train_examples)} with answers)")
    print(f"  Test:  {len(test_examples)} examples "
          f"({sum(e['has_answer'] for e in test_examples)} with answers)")

    # --- Step 1: Run full trees on train set (expensive, but only once) ---
    print("\n  [Tuning] Building full trees on training set (no pruning) ...")
    train_trees = []
    train_truths = []
    stats = RunStats()
    for i, ex in enumerate(train_examples):
        q_emb = train_q_embs[i : i + 1]
        root = TextNode(text=ex["document_text"], depth=0)
        # Run in "full" mode to get LLM labels for every leaf
        classify_node_recursive(
            root, ex["question"], q_emb, embedder,
            classifier, stats, 0.0, 0.0, "full",
        )
        # Now score every node with BERT (separate pass, no pruning)
        _score_tree_bert(root, q_emb, embedder, stats)
        train_trees.append(serialize_tree(root))
        train_truths.append("yes" if ex["has_answer"] else "no")

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(train_examples)} | "
                  f"LLM calls: {stats.llm_calls}")

    print(f"    Done. Total LLM calls: {stats.llm_calls}")

    # --- Step 2: Grid search on train trees (free) ---
    print("  [Tuning] Sweeping threshold parameters ...")
    base_values = [round(x, 2) for x in np.arange(0.20, 0.65, 0.05)]
    step_values = [round(x, 2) for x in np.arange(0.00, 0.25, 0.05)]
    doc_thresh_values = [0.0, 0.20, 0.35, 0.50]

    best_f1 = -1.0
    best_params = {}
    all_combos = []

    for base in base_values:
        for step in step_values:
            for doc_thresh in doc_thresh_values:
                preds = []
                total_llm = 0
                for tree, truth in zip(train_trees, train_truths):
                    label, _score, llm = replay_tree(
                        tree, base, step, doc_thresh
                    )
                    preds.append(label)
                    total_llm += llm

                report = classification_report(
                    train_truths, preds, labels=["yes", "no"],
                    output_dict=True, zero_division=0,
                )
                f1 = report["yes"]["f1-score"]
                acc = report["accuracy"]
                combo = {
                    "base_threshold": base,
                    "threshold_step": step,
                    "doc_score_threshold": doc_thresh,
                    "train_f1": f1,
                    "train_accuracy": acc,
                    "train_llm_calls": total_llm,
                }
                all_combos.append(combo)

                if f1 > best_f1 or (f1 == best_f1 and total_llm < best_params.get("train_llm_calls", float("inf"))):
                    best_f1 = f1
                    best_params = combo

    print(f"    Tested {len(all_combos)} combinations")
    print(f"    Best train F1: {best_params['train_f1']:.3f} "
          f"(acc: {best_params['train_accuracy']:.3f}, "
          f"LLM calls: {best_params['train_llm_calls']})")
    print(f"    Best params: base={best_params['base_threshold']:.2f}, "
          f"step={best_params['threshold_step']:.2f}, "
          f"doc_thresh={best_params['doc_score_threshold']:.2f}")

    # --- Step 3: Evaluate on test set with best params ---
    print(f"\n  [Tuning] Evaluating on test set with best params ...")
    test_result = run_evaluation(
        examples=test_examples,
        question_embeddings=test_q_embs,
        embedder=embedder,
        classifier=classifier,
        mode="bert",
        threshold=best_params["base_threshold"],
        threshold_step=best_params["threshold_step"],
        doc_score_threshold=best_params["doc_score_threshold"],
    )
    print_results(test_result)

    return {
        "best_params": best_params,
        "test_result": test_result,
        "all_combos": all_combos,
        "train_size": len(train_examples),
        "test_size": len(test_examples),
    }


def _score_tree_bert(node: TextNode, query_emb: np.ndarray,
                     embedder: BertEmbedder, stats: RunStats):
    """Walk a tree and add BERT scores to every node that fits in BERT's window."""
    words = node.text.strip().split()
    if len(words) <= BERT_MAX_WORDS and node.bert_score is None:
        emb = embedder.embed([node.text.strip()])
        stats.bert_calls += 1
        node.bert_score = float(
            embedder.cosine_similarity(query_emb, emb)[0]
        )
    for child in node.children:
        _score_tree_bert(child, query_emb, embedder, stats)


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
        help="Base BERT cosine similarity threshold for pruning (default: 0.40)",
    )
    parser.add_argument(
        "--threshold-step", type=float, default=THRESHOLD_STEP,
        help="Threshold increase per recursion depth (default: 0.10)",
    )
    parser.add_argument(
        "--aggregation", type=str, default="soft",
        choices=["soft", "hard"],
        help="Aggregation strategy: 'soft' (weight by BERT score) or 'hard' (any yes = yes)",
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
        help="Truncate documents to this many words (default: 0=no limit)",
    )
    parser.add_argument(
        "--local-data", type=str, default=None,
        help="Path to a local JSON file of pre-downloaded NQ examples "
             "(see test_data/download_nq_sample.py)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Tune thresholds using 80/20 train/test split before evaluation",
    )
    parser.add_argument(
        "--train-split", type=float, default=0.8,
        help="Fraction of data for training during --tune (default: 0.8)",
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

    # --- Threshold tuning ---
    if args.tune:
        tune_result = run_tuning(
            examples=examples,
            question_embeddings=question_embeddings,
            embedder=embedder,
            classifier=classifier,
            train_split=args.train_split,
            seed=args.seed,
        )
        # Apply tuned params to the main evaluation
        best = tune_result["best_params"]
        args.threshold = best["base_threshold"]
        args.threshold_step = best["threshold_step"]
        args.doc_score_threshold = best["doc_score_threshold"]
        print(f"\n  [Tuning] Applying tuned params: "
              f"base={args.threshold:.2f}, step={args.threshold_step:.2f}, "
              f"doc_thresh={args.doc_score_threshold:.2f}")

        # Save tuning results
        tune_output = {
            "best_params": best,
            "test_accuracy": tune_result["test_result"]["classification_report"]["accuracy"],
            "test_f1_yes": tune_result["test_result"]["classification_report"]["yes"]["f1-score"],
            "train_size": tune_result["train_size"],
            "test_size": tune_result["test_size"],
            "top_10_combos": sorted(
                tune_result["all_combos"],
                key=lambda c: (-c["train_f1"], c["train_llm_calls"]),
            )[:10],
        }
        tune_path = args.output.replace(".json", "_tuning.json")
        with open(tune_path, "w") as f:
            json.dump(tune_output, f, indent=2)
        print(f"  [Tuning] Results saved to {tune_path}")

    # Set default doc_score_threshold if not set by tuning
    if not hasattr(args, "doc_score_threshold"):
        args.doc_score_threshold = DOC_SCORE_THRESHOLD

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
            threshold_step=args.threshold_step,
            prune_rate=bert_prune_rate,
            aggregation=args.aggregation,
            doc_score_threshold=args.doc_score_threshold,
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
