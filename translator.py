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
from dotenv import load_dotenv

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a professional scientific translator. Translate each provided English sentence to concise,"
    " natural German. Preserve placeholders such as @ORGANISM$, @QUALITY$, @ENVIRONMENT$, punctuation,"
    " and any inline codes. Do not add explanations. Respond strictly in JSON with a top-level object"
    " {\"translations\": [..]} maintaining the original order."
)


def _extract_json_array(text: str) -> List[str]:
    """
    Parse a JSON array from a model response, tolerating stray text/code fences.
    Raises ValueError if parsing fails.
    """
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("translations", "output", "result"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
    except Exception:
        pass

    # Fallback: grab the first [...] segment.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        parsed = json.loads(snippet)
        if isinstance(parsed, list):
            return parsed

    raise ValueError("Response was not valid JSON array")


def _translate_batch_once(texts: List[str], model: str, api_key: str, timeout: int = 60) -> List[str]:
    """Single attempt to translate a list of sentences; raises on any issue."""
    user_prompt = {
        "role": "user",
        "content": (
            "Translate the following sentences to German. Respond ONLY with JSON: {\"translations\": [..]}.\n"
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
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    translations = _extract_json_array(content)
    if not isinstance(translations, list):
        raise ValueError("Unexpected translation payload (not a list)")

    # Be lenient: some models may return extra items; truncate if too long, but
    # still fail if too short to avoid silent data loss.
    if len(translations) < len(texts):
        raise ValueError(f"Unexpected translation length (got {len(translations)} for {len(texts)})")
    if len(translations) > len(texts):
        translations = translations[: len(texts)]

    return [str(t) for t in translations]


def translate_batch(texts: List[str], model: str, api_key: str, timeout: int = 60) -> List[str]:
    """Translate with up to 6 retries for transient errors."""
    for attempt in range(6):
        try:
            return _translate_batch_once(texts, model, api_key, timeout)
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt * 0.5
            sys.stderr.write(f"Batch failed (attempt {attempt+1}/6): {exc}. Retrying in {wait:.1f}s...\n")
            sys.stderr.flush()
            time.sleep(wait)
    raise RuntimeError("Translation failed after retries")


def translate_batch_with_fallback(texts: List[str], model: str, api_key: str, timeout: int = 60) -> List[str]:
    """
    Translate a list, falling back to per-sentence calls if the batch reply is malformed.
    """
    try:
        # Try once without consuming retries to avoid repeated length-mismatch noise.
        return _translate_batch_once(texts, model, api_key, timeout)
    except Exception as exc:
        sys.stderr.write(f"Batch parsing failed, falling back to per-sentence translation: {exc}\n")
        sys.stderr.flush()
        results: List[str] = []
        for idx, t in enumerate(texts):
            single = translate_batch([t], model, api_key, timeout)
            results.append(single[0])
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate CSV to German using OpenAI GPT.")
    parser.add_argument("--input", default="data/train.csv", help="Input CSV path (label,text)")
    parser.add_argument("--output", default="data/german_train.csv", help="Output CSV path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI chat model name")
    parser.add_argument("--batch-size", type=int, default=8, help="Rows per API call")
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Row index to start (0-based). If omitted with --resume, starts after existing output.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume automatically after the last translated row in --output (overrides missing --start).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to process")
    args = parser.parse_args()

    # Load API key from .env if present.
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Please set OPENAI_API_KEY in the environment.")

    # Determine starting index
    start_index = 0 if args.start is None else args.start
    if args.start is None and args.resume and os.path.exists(args.output):
        with open(args.output, newline="", encoding="utf-8") as f:
            start_index = sum(1 for _ in f)
        print(f"Resuming after {start_index} existing rows in {args.output}", file=sys.stderr)

    rows = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < start_index:
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
    mode = "a" if start_index > 0 and os.path.exists(out_path) else "w"

    with open(out_path, mode, newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        for idx in range(0, total, args.batch_size):
            batch = rows[idx : idx + args.batch_size]
            labels, texts = zip(*batch)
            translations = translate_batch_with_fallback(list(texts), args.model, api_key)
            for label, german in zip(labels, translations):
                writer.writerow([label, german])
            out_f.flush()
            done = min(idx + args.batch_size, total)
            print(f"Translated {done}/{total}", file=sys.stderr)

    print(f"Done. Wrote {total} rows to {out_path}")


if __name__ == "__main__":
    main()
