from __future__ import annotations

import os
import re
from collections.abc import Callable

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_gpt_response_with_relations(
    prompt_source: str | Callable[[str], str],
    text_df,
):
    text_samples = text_df["text"].tolist()
    extracted_labels = []
    for text in text_samples:
        prompt = prompt_source(text) if callable(prompt_source) else prompt_source
        messages = [
            {"role": "system", "content": "You are a helpful assistant for relation extraction."},
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Classify the following text and respond with the required format.\n"
                    f"Text: \"{text}\""
                ),
            },
        ]
        completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        response = completion.choices[0].message.content.strip()
        extracted_labels.append(response)
        print(extracted_labels)
    return extracted_labels

def parse_multiple_responses(responses):

    patterns = [
        re.compile(
            r"results?\s*=\s*\[\s*(?P<binary>[01])(?:\s*,\s*['\"]?(?P<label>[A-Za-z_]+)['\"]?)?\s*\]",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<binary>[01])\s*[,;]\s*['\"]?(?P<label>[A-Za-z_]+)['\"]?",
            re.IGNORECASE,
        ),
    ]

    parsed_rows = []
    for response in responses:
        raw_response = response if isinstance(response, str) else str(response)
        binary = None
        label = None

        for pattern in patterns:
            match = pattern.search(raw_response)
            if match:
                binary = int(match.group("binary"))
                label = match.group("label")
                break

        if binary is None:
            lone_match = re.search(r"\b([01])\b", raw_response)
            if lone_match:
                binary = int(lone_match.group(1))

        if binary is not None:
            if binary == 0:
                label = "NA"
            elif label:
                label = label.upper()
            else:
                label = "UNKNOWN"
        else:
            binary = -1
            label = "UNKNOWN"

        parsed_rows.append(
            {
                "raw_response": raw_response,
                "model_prediction": binary,
                "model_relation_label": label,
            }
        )

    return pd.DataFrame(parsed_rows)
