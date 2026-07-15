import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

NATURAL_PROMPT_FILE = Path("prompts/natural_language_prompt.txt")
CODE_PROMPT_PATHS = (Path("prompts/code_prompts.txt"),)

RELATION_CLASS_MAP = {
    "HAVE": "Have",
    "OCCUR_IN": "OccurIn",
    "INFLUENCE": "Influence",
}


def _load_base_prompt(path: Optional[Path] = None) -> str:
    target = path or NATURAL_PROMPT_FILE
    if target.exists():
        content = target.read_text(encoding="utf-8").strip()
        if content:
            return content
    raise FileNotFoundError(
        f"Natural-language prompt missing or empty at {target}. "
        "Please create the file with the instructions."
    )


def prompt_generator(examples_df, base_prompt_path: Optional[Path] = None):
    base_prompt = _load_base_prompt(base_prompt_path)
    prompt = [base_prompt]

    for idx, row in examples_df.iterrows():
        raw_label = str(row["gold"]).strip()
        if raw_label.upper() == "NA":
            label_text = "0, NA"
        else:
            label_text = f"1, {raw_label.upper()}"
        prompt.append(
            f'Example {idx + 1}:\n'
            f'Text: "{row["text"]}"\n'
            f"Gold label: {label_text}\n"
        )

    prompt.append("Classify the next sentence following the same output format.")
    return "\n".join(prompt)


def build_code_prompt_builder(
    template: str,
    *,
    entity_pair_selector=None,
    semantic_selector=None,
    log_recorder=None,
) -> Callable[[str], str]:
    placeholder = "{INPUT_TEXT}"

    def _builder(text: str) -> str:
        frames: list[pd.DataFrame] = []
        if entity_pair_selector is not None:
            dynamic_examples = entity_pair_selector.select_examples({"text": text})
            if dynamic_examples:
                frames.append(
                    pd.DataFrame(dynamic_examples).assign(_source="entity_pair")
                )
        if semantic_selector is not None:
            semantic_examples = semantic_selector.select_examples({"text": text})
            if semantic_examples:
                frames.append(
                    pd.DataFrame(semantic_examples).assign(_source="semantic")
                )

        if not frames:
            combined_df = pd.DataFrame(columns=["gold", "text"])
        elif len(frames) == 1:
            combined_df = frames[0]
        else:
            combined_df = pd.concat(frames, ignore_index=True)

        if log_recorder is not None:
            log_recorder(text, combined_df.copy())

        snippets: list[str] = []
        for _, row in combined_df.iterrows():
            label = str(row["gold"]).strip().upper()
            example_text = str(row["text"]).strip()
            class_name = RELATION_CLASS_MAP.get(label)
            if not label or label == "NA" or class_name is None:
                snippets.append(
                    "# Few-shot Example (NA)\n"
                    f"context = {json.dumps(example_text)}\n"
                    'reasoning = "no explicit relation"\n'
                    "results = [0]"
                )
            else:
                snippets.append(
                    f"# Few-shot Example ({label})\n"
                    f"context = {json.dumps(example_text)}\n"
                    f'reasoning = "{label} example"\n'
                    f"results = [1, {class_name}]"
                )

        if not snippets:
            snippets.append(
                "# Few-shot Example (NA)\n"
                '# (no dynamic examples available)\n'
                'reasoning = "no explicit relation"\n'
                "results = [0]"
            )

        few_shot_section = "\n\n".join(snippets)

        if placeholder in template:
            prompt_body = template.replace(placeholder, text)
        else:
            escaped = json.dumps(text)
            prompt_body = (
                f"{template.rstrip()}\n\n# =========================\n"
                f"# Input\n# =========================\ncontext = {escaped}\n"
            )

        return (
            f"{prompt_body}\n\n# =========================\n"
            "# Few-shot Examples\n# =========================\n"
            f"{few_shot_section}"
        )

    return _builder


def build_zero_shot_code_prompt_builder(template: str) -> Callable[[str], str]:
    """
    Build a prompt builder that uses the code-style template in pure zero-shot
    mode (no few-shot examples are appended).
    """
    placeholder = "{INPUT_TEXT}"

    def _builder(text: str) -> str:
        if placeholder in template:
            return template.replace(placeholder, text)
        escaped = json.dumps(text)
        return (
            f"{template.rstrip()}\n\n# =========================\n"
            f"# Input\n# =========================\ncontext = {escaped}\n"
        )

    return _builder
