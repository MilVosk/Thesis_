#!/usr/bin/env python3
"""
Translate data/train.csv to data/german_train.csv using OpenAI's Chat Completions API.
- Keeps the first column (label) unchanged.
- Translates the text column to German.
- Batches requests to keep token usage reasonable.

Usage (requires env OPENAI_API_KEY):
    python3 translator.py --input data/train.csv --output data/german_train.csv \
        --model gpt-4o-mini --batch-size 8

This script uses only the standard library plus `requests`. Install requests if missing:
    python3 -m pip install --user requests
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from typing import List

import requests

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a professional scientific translator. Translate each provided English sentence to concise,"
    " natural German. Preserve placeholders such as @ORGANISM$, @QUALITY$, @ENVIRONMENT$, punctuation,"
    " and any inline codes. Do not add explanations. Return a JSON array of translated strings in the"
    " same order as given."
)


def translate_batch(texts: List[str], model: str, api_key: str, timeout: int = 60) -> List[str]:
    """Translate a list of sentences, returning a list of German strings."""
    user_prompt = {
        "role": "user",
        "content": (
            "Translate the following sentences to German. Respond ONLY with a JSON array of strings.\n"
            "Sentences:\n" + "\n".join(f"- {t}" for t in texts)
        ),
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_prompt,
        ],
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(6):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            translations = json.loads(content)
            if not isinstance(translations, list) or len(translations) != len(texts):
                raise ValueError("Unexpected translation format/length")
            return [str(t) for t in translations]
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt * 0.5
            sys.stderr.write(f"Batch failed (attempt {attempt+1}/6): {exc}. Retrying in {wait:.1f}s...\n")
            sys.stderr.flush()
            time.sleep(wait)
    raise RuntimeError("Translation failed after retries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate CSV to German using OpenAI GPT.")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV path (label,text)")
    parser.add_argument("--output", default="data/german_train.csv", help="Output CSV path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI chat model name")
    parser.add_argument("--batch-size", type=int, default=8, help="Rows per API call")
    parser.add_argument("--start", type=int, default=0, help="Row index to start (for resumes)")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to process")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Please set OPENAI_API_KEY in the environment.")

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < args.start:
                continue
            if args.limit is not None and len(rows) >= args.limit:
                break
            if len(row) < 2:
                sys.stderr.write(f"Skipping malformed row {i}\n")
                continue
            rows.append((row[0], row[1]))

    total = len(rows)
    if total == 0:
        sys.exit("No rows to translate.")

    out_path = args.output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        for idx in range(0, total, args.batch_size):
            batch = rows[idx : idx + args.batch_size]
            labels, texts = zip(*batch)
            translations = translate_batch(list(texts), args.model, api_key)
            for label, german in zip(labels, translations):
                writer.writerow([label, german])
            out_f.flush()
            done = min(idx + args.batch_size, total)
            print(f"Translated {done}/{total}", file=sys.stderr)

    print(f"Done. Wrote {total} rows to {out_path}")


if __name__ == "__main__":
    main()
