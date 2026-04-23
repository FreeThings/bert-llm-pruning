"""
BERT-Augmented LLM Text Classification on Natural Questions
CS4964 Final Project — Greenbaum, McFarland, DeBoer

Algorithm (gate-then-chunk with depth-scaled thresholds):
  1. Load NQ dataset
  2. Split each document into BERT-max windows (~380 words)
     (BERT can only see 512 tokens, so scoring full docs is unreliable)
  3. BERT-score each window against the query, prune below threshold
  4. Split survivors into paragraphs, score and prune (tighter threshold)
  5. Surviving leaf chunks sent to Llama in a single batched call per document
  6. Aggregate: soft (weighted mean of bert scores) or hard (any yes = yes)
  Threshold scales with depth: τ(d) = base + d * step

Evaluation compares three modes:
  - full    : LLM called on every document (no pruning)
  - bert    : BERT-guided pruning (our method)
  - random  : random pruning at the same rate as bert (control)
"""

import argparse
import json
import os
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
# sentence-transformers model: trained for semantic similarity, much better
# cosine scores than bert-base-uncased for question-passage matching
BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.23   # base cosine similarity gate for pruning (tuned)
THRESHOLD_STEP = 0.08         # increase threshold by this much per depth level (tuned)
BERT_MAX_WORDS = 380           # ~512 tokens — max BERT can actually see
MIN_CHUNK_WORDS = 40           # stop recursing — "few sentences" leaf size
MAX_RECURSION_DEPTH = 3
MAX_DOC_WORDS = 0             # 0 = no truncation
DOC_SCORE_THRESHOLD = 0.0     # soft aggregation threshold (unused in hard mode)

# Chunk sizes per depth level
CHUNK_WORDS_BY_DEPTH = [BERT_MAX_WORDS, 150, 40]

# Sliding window overlap per depth level — wider at coarse levels to avoid
# boundary misses; narrower at leaf level where chunks are already small
CHUNK_OVERLAP_BY_DEPTH = [50, 30, 10]

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

    def _make_prompt(self, query: str, text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a precise text classifier. Answer only with 'yes' or 'no'."},
            {"role": "user", "content": (
                f"Does the following passage answer or address this question?\n\n"
                f"Question: {query}\n\n"
                f"Passage: {text}\n\n"
                f"Answer with only 'yes' or 'no'."
            )},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _parse_answer(self, result: str, prompt: str) -> str:
        answer = result[len(prompt):].strip().lower()
        if answer.startswith("yes"):
            return "yes"
        if answer.startswith("no"):
            return "no"
        return "yes" if "yes" in answer else "no"

    def classify_batch(self, query: str, texts: list[str]) -> list[str]:
        """Classify a batch of texts in a single pipeline call."""
        if not texts:
            return []
        prompts = [self._make_prompt(query, t) for t in texts]
        results = self.pipe(prompts, batch_size=min(8, len(prompts)))
        return [
            self._parse_answer(r[0]["generated_text"], p)
            for r, p in zip(results, prompts)
        ]


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def get_chunk_size_for_depth(depth: int) -> int:
    if depth < len(CHUNK_WORDS_BY_DEPTH):
        return CHUNK_WORDS_BY_DEPTH[depth]
    return MIN_CHUNK_WORDS


def get_overlap_for_depth(depth: int) -> int:
    if depth < len(CHUNK_OVERLAP_BY_DEPTH):
        return CHUNK_OVERLAP_BY_DEPTH[depth]
    return CHUNK_OVERLAP_BY_DEPTH[-1]


def split_into_chunks(text: str, max_words: int, overlap: int) -> list[str]:
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
# Two-phase document evaluation
# ---------------------------------------------------------------------------

def _gate_and_chunk(
    node: TextNode,
    bert_query_embedding: np.ndarray,
    embedder: BertEmbedder,
    stats: RunStats,
    base_threshold: float,
    threshold_step: float,
    mode: str,
    prune_rate: float,
    gates_passed: int,
    pending_leaves: list,
):
    """
    Phase 1: Apply BERT gating and recursive chunking.
    Surviving leaf nodes are appended to pending_leaves; no LLM calls made here.

    Threshold tightens per gate passed: τ = base + gates_passed * step.
    Oversized nodes (> BERT_MAX_WORDS) skip the gate; first real gate is depth 1.
    """
    stats.total_nodes += 1
    text = node.text.strip()
    words = text.split()
    is_oversized = len(words) > BERT_MAX_WORDS
    depth_threshold = base_threshold + gates_passed * threshold_step

    # --- Gate ---
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
                return
        elif mode == "random":
            if random.random() < prune_rate:
                node.was_pruned = True
                stats.pruned_nodes += 1
                return
        gates_passed += 1

    # --- Leaf: queue for LLM ---
    if len(words) <= MIN_CHUNK_WORDS or node.depth >= MAX_RECURSION_DEPTH:
        pending_leaves.append(node)
        return

    # --- Recurse: split into depth-appropriate chunks ---
    child_chunk_size = get_chunk_size_for_depth(node.depth)
    overlap = get_overlap_for_depth(node.depth)
    chunks = split_into_chunks(text, max_words=child_chunk_size, overlap=overlap)

    if len(chunks) == 1:
        pending_leaves.append(node)
        return

    children = [TextNode(text=ct, depth=node.depth + 1) for ct in chunks]
    node.children = children

    # Batch BERT scoring for children (groups of 32 to avoid OOM)
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

    for child in children:
        _gate_and_chunk(
            child, bert_query_embedding, embedder, stats,
            base_threshold, threshold_step, mode, prune_rate, gates_passed,
            pending_leaves,
        )


def _aggregate_tree(
    node: TextNode,
    aggregation: str,
    doc_score_threshold: float,
    base_threshold: float,
) -> tuple[str, float]:
    """
    Phase 3: Aggregate LLM labels up the tree after all leaves are classified.

    Soft mode uses a weighted mean of yes-scores across children so both
    signal quality (bert score) and vote fraction contribute to the decision.
    Oversized nodes without a bert_score fall back to base_threshold as their
    contribution weight instead of 0.0, avoiding silent false negatives.
    """
    if not node.children:
        if node.was_pruned or node.llm_label is None:
            return "no", 0.0
        label = node.llm_label
        score = (
            (node.bert_score if node.bert_score is not None else base_threshold)
            if label == "yes" else 0.0
        )
        return label, score

    child_scores = []
    child_labels = []
    for child in node.children:
        label, score = _aggregate_tree(child, aggregation, doc_score_threshold, base_threshold)
        child_scores.append(score)
        child_labels.append(label)

    if aggregation == "hard":
        result = "yes" if "yes" in child_labels else "no"
        agg_score = max(child_scores) if child_scores else 0.0
    else:
        n = len(child_scores) or 1
        agg_score = sum(child_scores) / n
        result = "yes" if agg_score > doc_score_threshold else "no"

    node.llm_label = result
    return result, agg_score


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

        annotations = row["annotations"]
        long_answers = annotations["long_answer"]
        has_answer = any(la["start_token"] >= 0 for la in long_answers)

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
    """Load pre-downloaded examples from a local JSON file."""
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
    aggregation: str = "hard",
    doc_score_threshold: float = DOC_SCORE_THRESHOLD,
    checkpoint_path: str = None,
) -> dict:
    """
    Run classification on all examples under the given mode.

    Phase 1: _gate_and_chunk — BERT gating + chunking, collects surviving leaves.
    Phase 2: classify_batch — single batched LLM call per document.
    Phase 3: _aggregate_tree — merge labels up the tree.
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
    elapsed_before_resume = 0.0
    predictions = []
    ground_truths = []
    start_idx = 0

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        if ckpt.get("mode") == mode:
            predictions = ckpt["predictions"]
            ground_truths = ckpt["ground_truths"]
            stats.llm_calls = ckpt["llm_calls"]
            stats.bert_calls = ckpt["bert_calls"]
            stats.pruned_nodes = ckpt["pruned_nodes"]
            stats.total_nodes = ckpt["total_nodes"]
            elapsed_before_resume = ckpt.get("elapsed", 0.0)
            start_idx = len(predictions)
            print(f"  [Checkpoint] Resuming from example {start_idx}/{len(examples)}")

    for i, ex in enumerate(examples[start_idx:], start=start_idx):
        q_emb = question_embeddings[i : i + 1]

        # Phase 1
        root = TextNode(text=ex["document_text"], depth=0)
        pending_leaves: list[TextNode] = []
        _gate_and_chunk(
            root, q_emb, embedder, stats,
            threshold, threshold_step, mode, prune_rate, 0, pending_leaves,
        )

        # Phase 2
        if pending_leaves:
            texts = [leaf.text for leaf in pending_leaves]
            labels = classifier.classify_batch(ex["question"], texts)
            for leaf, lbl in zip(pending_leaves, labels):
                leaf.llm_label = lbl
            stats.llm_calls += len(pending_leaves)

        # Phase 3
        label, _score = _aggregate_tree(root, aggregation, doc_score_threshold, threshold)
        predictions.append(label)
        ground_truths.append("yes" if ex["has_answer"] else "no")

        if (i + 1) % 10 == 0:
            print(f"  [{mode}] Processed {i+1}/{len(examples)} | "
                  f"LLM calls so far: {stats.llm_calls}")

        if checkpoint_path and (i + 1) % 20 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "mode": mode,
                    "predictions": predictions,
                    "ground_truths": ground_truths,
                    "llm_calls": stats.llm_calls,
                    "bert_calls": stats.bert_calls,
                    "pruned_nodes": stats.pruned_nodes,
                    "total_nodes": stats.total_nodes,
                    "elapsed": elapsed_before_resume + (time.time() - t0),
                }, f)

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    stats.wall_time = elapsed_before_resume + (time.time() - t0)

    report = classification_report(
        ground_truths, predictions, labels=["yes", "no"],
        output_dict=True, zero_division=0,
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


def replay_tree(
    node_data: dict,
    base_threshold: float,
    threshold_step: float,
    doc_score_threshold: float,
    aggregation: str = "hard",
    gates_passed: int = 0,
) -> tuple[str, float, int]:
    """
    Simulate pruning on a pre-scored tree. Returns (label, score, llm_calls).
    No model calls — replays threshold logic on cached scores.
    Mirrors _aggregate_tree: soft uses weighted mean, oversized nodes use
    base_threshold as fallback score.
    """
    bert_score = node_data["bert_score"]
    depth_threshold = base_threshold + gates_passed * threshold_step

    if bert_score is not None:
        if bert_score < depth_threshold:
            return "no", 0.0, 0
        gates_passed += 1

    if node_data["is_leaf"]:
        label = node_data["llm_label"]
        score = (
            (bert_score if bert_score is not None else base_threshold)
            if label == "yes" else 0.0
        )
        return label, score, 1

    child_scores = []
    child_labels = []
    total_llm = 0
    for child in node_data["children"]:
        label, score, llm = replay_tree(
            child, base_threshold, threshold_step, doc_score_threshold,
            aggregation, gates_passed,
        )
        child_scores.append(score)
        child_labels.append(label)
        total_llm += llm

    if aggregation == "hard":
        result = "yes" if "yes" in child_labels else "no"
        agg_score = max(child_scores) if child_scores else 0.0
    else:
        n = len(child_scores) or 1
        agg_score = sum(child_scores) / n
        result = "yes" if agg_score > doc_score_threshold else "no"

    return result, agg_score, total_llm


def run_tuning(
    examples: list[dict],
    question_embeddings: np.ndarray,
    embedder: BertEmbedder,
    classifier: LlamaClassifier,
    train_split: float = 0.8,
    seed: int = 42,
    aggregation: str = "hard",
) -> dict:
    """
    Tune thresholds using a stratified train/test split.

    1. Split examples into train/test
    2. Run full tree (no pruning) on train set to collect BERT scores + LLM labels
    3. Coarse grid sweep on train trees (free — no model calls)
    4. Fine-pass sweep ±0.04 around the coarse optimum
    5. Pick best params by F1, break ties by fewest LLM calls
    6. Evaluate on test set with best params
    """
    rng = random.Random(seed)

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

    # --- Build full trees on train set (batched LLM + separate BERT pass) ---
    print("\n  [Tuning] Building full trees on training set ...")
    train_trees = []
    train_truths = []
    stats = RunStats()
    for i, ex in enumerate(train_examples):
        q_emb = train_q_embs[i : i + 1]
        root = TextNode(text=ex["document_text"], depth=0)

        pending_leaves: list[TextNode] = []
        _gate_and_chunk(
            root, q_emb, embedder, stats,
            0.0, 0.0, "full", 0.0, 0, pending_leaves,
        )
        if pending_leaves:
            texts = [l.text for l in pending_leaves]
            labels = classifier.classify_batch(ex["question"], texts)
            for leaf, lbl in zip(pending_leaves, labels):
                leaf.llm_label = lbl
            stats.llm_calls += len(pending_leaves)

        _score_tree_bert(root, q_emb, embedder, stats)
        train_trees.append(serialize_tree(root))
        train_truths.append("yes" if ex["has_answer"] else "no")

        if (i + 1) % 10 == 0:
            print(f"    Processed {i+1}/{len(train_examples)} | "
                  f"LLM calls: {stats.llm_calls}")

    print(f"    Done. Total LLM calls: {stats.llm_calls}")

    # --- Shared sweep helper ---
    def _sweep(base_values, step_values, doc_thresh_values):
        best_f1 = -1.0
        best_params: dict = {}
        combos = []
        for base in base_values:
            for step in step_values:
                for doc_thresh in doc_thresh_values:
                    preds = []
                    total_llm = 0
                    for tree, _ in zip(train_trees, train_truths):
                        label, _score, llm = replay_tree(
                            tree, base, step, doc_thresh, aggregation
                        )
                        preds.append(label)
                        total_llm += llm
                    report = classification_report(
                        train_truths, preds, labels=["yes", "no"],
                        output_dict=True, zero_division=0,
                    )
                    f1 = report["yes"]["f1-score"]
                    combo = {
                        "base_threshold": round(float(base), 4),
                        "threshold_step": round(float(step), 4),
                        "doc_score_threshold": round(float(doc_thresh), 4),
                        "train_f1": f1,
                        "train_accuracy": report["accuracy"],
                        "train_llm_calls": total_llm,
                    }
                    combos.append(combo)
                    if f1 > best_f1 or (
                        f1 == best_f1
                        and total_llm < best_params.get("train_llm_calls", float("inf"))
                    ):
                        best_f1 = f1
                        best_params = combo
        return best_params, combos

    # --- Coarse pass ---
    print("  [Tuning] Coarse grid search ...")
    base_coarse = [round(x, 2) for x in np.arange(0.20, 0.65, 0.05)]
    step_coarse = [round(x, 2) for x in np.arange(0.00, 0.30, 0.05)]
    doc_coarse  = [0.0] if aggregation == "hard" else [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    best_coarse, all_combos = _sweep(base_coarse, step_coarse, doc_coarse)
    print(f"    {len(all_combos)} combos | best coarse F1: {best_coarse['train_f1']:.3f} "
          f"base={best_coarse['base_threshold']:.2f} "
          f"step={best_coarse['threshold_step']:.2f} "
          f"doc={best_coarse['doc_score_threshold']:.2f}")

    # --- Fine pass: ±0.08 around coarse optimum at 0.02 resolution ---
    print("  [Tuning] Fine-pass search around best region ...")
    b0 = best_coarse["base_threshold"]
    s0 = best_coarse["threshold_step"]
    d0 = best_coarse["doc_score_threshold"]
    base_fine = [round(x, 3) for x in np.arange(max(0.05, b0 - 0.08), min(0.95, b0 + 0.09), 0.02)]
    step_fine = [round(x, 3) for x in np.arange(max(0.00, s0 - 0.08), min(0.50, s0 + 0.09), 0.02)]
    doc_fine  = [0.0] if aggregation == "hard" else [round(x, 3) for x in np.arange(max(0.00, d0 - 0.08), min(0.90, d0 + 0.09), 0.02)]
    best_fine, fine_combos = _sweep(base_fine, step_fine, doc_fine)
    all_combos.extend(fine_combos)
    print(f"    {len(fine_combos)} fine combos | best fine F1: {best_fine['train_f1']:.3f} "
          f"base={best_fine['base_threshold']:.3f} "
          f"step={best_fine['threshold_step']:.3f} "
          f"doc={best_fine['doc_score_threshold']:.3f}")

    best_params = (
        best_fine if best_fine["train_f1"] >= best_coarse["train_f1"] else best_coarse
    )
    print(f"  [Tuning] Final best: F1={best_params['train_f1']:.3f} "
          f"acc={best_params['train_accuracy']:.3f} "
          f"LLM={best_params['train_llm_calls']} | "
          f"base={best_params['base_threshold']:.3f} "
          f"step={best_params['threshold_step']:.3f} "
          f"doc={best_params['doc_score_threshold']:.3f}")

    # --- Test evaluation ---
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
        aggregation=aggregation,
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
    pending = []
    def _collect(n):
        if len(n.text.strip().split()) <= BERT_MAX_WORDS and n.bert_score is None:
            pending.append(n)
        for c in n.children:
            _collect(c)
    _collect(node)

    if not pending:
        return

    texts = [n.text.strip() for n in pending]
    emb_batches = [
        embedder.embed(texts[i : i + 32])
        for i in range(0, len(texts), 32)
    ]
    embs = np.concatenate(emb_batches, axis=0)
    sims = embedder.cosine_similarity(query_emb, embs)
    for n, sim in zip(pending, sims):
        n.bert_score = float(sim)
    stats.bert_calls += len(pending)


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
        help="Number of NQ examples (default: 50; use ≥150 for reliable tuning)",
    )
    parser.add_argument(
        "--split", type=str, default="validation",
        choices=["train", "validation"],
    )
    parser.add_argument(
        "--threshold", type=float, default=SIMILARITY_THRESHOLD,
        help="Base BERT cosine similarity threshold for pruning (default: 0.35)",
    )
    parser.add_argument(
        "--threshold-step", type=float, default=THRESHOLD_STEP,
        help="Threshold increase per recursion depth (default: 0.20)",
    )
    parser.add_argument(
        "--aggregation", type=str, default="hard",
        choices=["soft", "hard"],
        help="'hard' = any yes → yes (default); 'soft' = weighted mean of BERT scores",
    )
    parser.add_argument("--llama-model", type=str, default=None)
    parser.add_argument("--tiny", action="store_true",
                        help="Use TinyLlama (no HF auth required)")
    parser.add_argument(
        "--modes", nargs="+", default=["full", "bert", "random"],
        choices=["full", "bert", "random"],
    )
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument(
        "--max-doc-words", type=int, default=MAX_DOC_WORDS,
        help="Truncate documents to this many words (0 = no limit)",
    )
    parser.add_argument(
        "--doc-score-threshold", type=float, default=DOC_SCORE_THRESHOLD,
        help="Soft aggregation threshold (default: 0.10; unused in hard mode)",
    )
    parser.add_argument("--local-data", type=str, default=None,
                        help="Path to local JSON of pre-downloaded NQ examples")
    parser.add_argument("--save-data", type=str, default=None,
                        help="Save loaded examples to this JSON path for offline reuse")
    parser.add_argument("--tune", action="store_true",
                        help="Tune thresholds via coarse+fine grid search before evaluation")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.llama_model:
        llm_model = args.llama_model
    elif args.tiny:
        llm_model = TINYLLAMA_MODEL
    else:
        llm_model = DEFAULT_LLAMA_MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using: {device}")
    print(f"[LLM]   Model: {llm_model}")

    embedder = BertEmbedder(device=device)
    classifier = LlamaClassifier(model_name=llm_model, device=device)

    if args.local_data:
        examples = load_local_examples(args.local_data, n=args.n)
    else:
        examples = load_nq_examples(
            n=args.n, split=args.split, max_doc_words=args.max_doc_words or None,
        )

    if args.save_data:
        with open(args.save_data, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"[NQ] Examples saved to {args.save_data}")

    print("[BERT] Pre-computing question embeddings ...")
    questions = [ex["question"] for ex in examples]
    q_batches = [
        embedder.embed(questions[i : i + 64])
        for i in range(0, len(questions), 64)
    ]
    question_embeddings = np.concatenate(q_batches, axis=0)

    if args.tune:
        tune_result = run_tuning(
            examples=examples,
            question_embeddings=question_embeddings,
            embedder=embedder,
            classifier=classifier,
            train_split=args.train_split,
            seed=args.seed,
            aggregation=args.aggregation,
        )
        best = tune_result["best_params"]
        args.threshold = best["base_threshold"]
        args.threshold_step = best["threshold_step"]
        args.doc_score_threshold = best["doc_score_threshold"]
        print(f"\n  [Tuning] Applying tuned params: "
              f"base={args.threshold:.3f}, step={args.threshold_step:.3f}, "
              f"doc_thresh={args.doc_score_threshold:.3f}")

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

    modes = list(args.modes)
    if "bert" in modes and "random" in modes:
        modes.sort(key=lambda m: 0 if m == "full" else (1 if m == "bert" else 2))

    all_results = []
    bert_prune_rate = 0.0
    if "random" in modes and "bert" not in modes:
        print("[Warning] 'random' mode requested without 'bert' — prune rate "
              "defaults to 0.0. Add 'bert' to --modes for a matched comparison.")

    for mode in modes:
        ckpt_path = args.output.replace(".json", f".{mode}.ckpt")
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
            checkpoint_path=ckpt_path,
        )
        if mode == "bert":
            s = result["stats"]
            bert_prune_rate = s.pruned_nodes / s.total_nodes if s.total_nodes > 0 else 0.0
            print(f"  [bert] Actual prune rate: {bert_prune_rate:.3f}")
        print_results(result)
        all_results.append(result)

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
