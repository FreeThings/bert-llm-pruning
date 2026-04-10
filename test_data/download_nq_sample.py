"""
Download a random sample of Natural Questions examples and save locally.

Usage:
    python test_data/download_nq_sample.py [--n 50] [--split validation] [--seed 42]

Output:
    test_data/nq_sample_50.json  (or whatever --n is set to)

This is slow the first time (NQ streams full Wikipedia pages), but only
needs to be run once.  After that, use:
    python classify_nq.py --local-data test_data/nq_sample_50.json --tiny ...
"""

import argparse
import json
import random
import sys
import time

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download NQ sample to local JSON")
    parser.add_argument("--n", type=int, default=50, help="Number of examples")
    parser.add_argument("--split", type=str, default="validation",
                        choices=["train", "validation"])
    parser.add_argument("--max-doc-words", type=int, default=500,
                        help="Truncate documents to this many words (0=no limit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: test_data/nq_sample_{n}.json)")
    args = parser.parse_args()

    random.seed(args.seed)
    output = args.output or f"test_data/nq_sample_{args.n}.json"

    print(f"Streaming NQ {args.split} split (this may take a few minutes) ...")
    t0 = time.time()

    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split=args.split,
        streaming=True,
    )

    examples = []
    for i, row in enumerate(ds):
        question = row["question"]["text"]

        doc_tokens = row["document"]["tokens"]
        token_list = doc_tokens["token"]
        is_html_list = doc_tokens["is_html"]
        plain_words = [
            tok for tok, is_html in zip(token_list, is_html_list)
            if not is_html
        ]
        if args.max_doc_words:
            plain_words = plain_words[:args.max_doc_words]
        document_text = " ".join(plain_words)

        annotations = row["annotations"]
        long_answers = annotations["long_answer"]
        has_answer = any(la["start_token"] >= 0 for la in long_answers)

        examples.append({
            "question": question,
            "document_text": document_text,
            "has_answer": has_answer,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Downloaded {i + 1} examples ({elapsed:.0f}s elapsed)")

        if len(examples) >= args.n:
            break

    elapsed = time.time() - t0
    n_pos = sum(e["has_answer"] for e in examples)
    print(f"Done: {len(examples)} examples ({n_pos} with answers) in {elapsed:.0f}s")

    with open(output, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
