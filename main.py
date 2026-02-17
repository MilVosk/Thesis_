import csv
import json
from typing import Callable, Optional

import pandas as pd

from paths import (
    CODE_PROMPT_FILE,
    CODE_PROMPTS_FILE,
    FEW_SHOT_LOG,
    PROMPT_PREVIEW_FILE,
    RESULTS_CSV,
    ensure_directories,
)
from utils.data_loader import get_dataframe
from utils.gpt_utils import generate_gpt_response_with_relations, parse_multiple_responses
from utils.langchain_shot_selector import (
    build_balanced_entity_pair_selector,
    build_semantic_similarity_selector,
)
from utils.prompt_generator import prompt_generator


EVAL_CSV_PATH = "data/test.csv"
EVAL_HAS_HEADER = True
# Set to None to evaluate on the full test set.
EVAL_ROW_LIMIT = None
CODE_PROMPT_PATHS = (CODE_PROMPT_FILE, CODE_PROMPTS_FILE)

# Prompt / evaluation configuration
# Set USE_ZERO_SHOT = True to run pure zero-shot classification (no few-shot examples).
USE_ZERO_SHOT = False
# Toggle to enable the structured code prompt template instead of the natural-language prompt.
USE_CODE_PROMPT = False

# Controls how many dynamic examples are selected around the current sentence
# when using the balanced entity-pair selector (few-shot mode only).
DYNAMIC_POSITIVE_SAMPLES = 2
DYNAMIC_NA_SAMPLES = 4

# Controls semantic-similarity retrieval for dynamic few-shot prompts.
USE_SEMANTIC_SELECTOR = True
SEMANTIC_SIMILARITY_SAMPLES = 4

RELATION_CLASS_MAP = {
    "HAVE": "Have",
    "OCCUR_IN": "OccurIn",
    "INFLUENCE": "Influence",
}

def build_prompt_builder(
    base_examples_df: Optional[pd.DataFrame],
    *,
    entity_pair_selector=None,
    semantic_selector=None,
    na_selector=None,
    positive_selector=None,
    log_recorder=None,
):
    def _builder(text: str) -> str:
        frames: list[pd.DataFrame] = []
        if base_examples_df is not None and not base_examples_df.empty:
            frames.append(base_examples_df.assign(_source="base"))

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

        if positive_selector is not None:
            positive_examples = positive_selector.select_examples({"text": text})
            if positive_examples:
                frames.append(
                    pd.DataFrame(positive_examples).assign(_source="positive")
                )

        if na_selector is not None:
            na_examples = na_selector.select_examples({"text": text})
            if na_examples:
                frames.append(pd.DataFrame(na_examples).assign(_source="na"))

        if not frames:
            combined_df = pd.DataFrame(columns=["gold", "text"])
        elif len(frames) == 1:
            combined_df = frames[0]
        else:
            combined_df = pd.concat(frames, ignore_index=True)

        if log_recorder is not None:
            log_recorder(text, combined_df.copy())

        prompt_df = combined_df.drop(columns=["_source"], errors="ignore")
        return prompt_generator(prompt_df)

    return _builder


def build_code_prompt_builder(
    template: str,
    *,
    base_examples_df: Optional[pd.DataFrame] = None,
    entity_pair_selector=None,
    semantic_selector=None,
    log_recorder=None,
) -> Callable[[str], str]:
    placeholder = "{INPUT_TEXT}"

    def _builder(text: str) -> str:
        frames: list[pd.DataFrame] = []
        if base_examples_df is not None and not base_examples_df.empty:
            frames.append(base_examples_df.assign(_source="base"))

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


def load_code_prompt() -> Optional[str]:
    for candidate in CODE_PROMPT_PATHS:
        if candidate is None:
            continue
        if not candidate.exists():
            continue
        content = candidate.read_text(encoding="utf-8").strip()
        if content:
            return content
    return None


def main() -> None:
    ensure_directories()
    code_prompt_template = load_code_prompt() if USE_CODE_PROMPT else None
    few_shot_logs: list[pd.DataFrame] = []

    def record_few_shot_examples(input_text: str, examples_df: pd.DataFrame) -> None:
        log_df = examples_df.copy()
        log_df["input_text"] = input_text
        few_shot_logs.append(log_df)

    # Prepare examples and prompt builder depending on zero-shot vs few-shot mode.
    if USE_ZERO_SHOT:
        # Pure zero-shot: no labeled examples are used.
        shot_df = pd.DataFrame(columns=["gold", "text"])
        entity_pair_selector = None

        # Still write a prompt preview (instructions only) for inspection.
        prompt_preview = prompt_generator(shot_df)
        PROMPT_PREVIEW_FILE.write_text(prompt_preview, encoding="utf-8")

        if code_prompt_template:
            prompt_builder = build_zero_shot_code_prompt_builder(code_prompt_template)
        else:
            # Simple zero-shot builder: always use an empty examples DataFrame.
            def prompt_builder(text: str) -> str:  # type: ignore[assignment]
                _ = text
                empty_df = pd.DataFrame(columns=["gold", "text"])
                return prompt_generator(empty_df)
    else:
        # Few-shot mode: rely exclusively on dynamic selectors (no static base pool).
        base_examples_df = None

        semantic_selector = None
        try:
            entity_pair_selector = build_balanced_entity_pair_selector(
                source_csv="data/train.csv",
                label_column="gold",
                text_column="text",
                positive_samples=DYNAMIC_POSITIVE_SAMPLES,
                na_samples=DYNAMIC_NA_SAMPLES,
                has_header=False,
            )
        except ValueError as exc:
            print(
                "Warning: unable to build balanced entity-pair selector "
                f"({exc}). Falling back to static few-shot examples."
            )
            entity_pair_selector = None

        if USE_SEMANTIC_SELECTOR:
            try:
                semantic_selector = build_semantic_similarity_selector(
                    source_csv="data/train.csv",
                    label_column="gold",
                    text_column="text",
                    has_header=False,
                    top_k=SEMANTIC_SIMILARITY_SAMPLES,
                )
            except (ImportError, ValueError) as exc:
                print(
                    "Warning: unable to build semantic similarity selector "
                    f"({exc}). Continuing without semantic examples."
                )
                semantic_selector = None

        if code_prompt_template:
            prompt_builder = build_code_prompt_builder(
                code_prompt_template,
                base_examples_df=base_examples_df,
                entity_pair_selector=entity_pair_selector,
                semantic_selector=semantic_selector,
                log_recorder=record_few_shot_examples,
            )
        else:
            prompt_builder = build_prompt_builder(
                base_examples_df,
                entity_pair_selector=entity_pair_selector,
                semantic_selector=semantic_selector,
                log_recorder=record_few_shot_examples,
            )

    eval_df = get_dataframe(
        EVAL_CSV_PATH,
        columns=None,
        has_header=EVAL_HAS_HEADER,
        keep_default_na=False,
    )
    if "text" not in eval_df.columns:
        raise ValueError(f"{EVAL_CSV_PATH} must contain a 'text' column for inference.")
    if "gold" not in eval_df.columns:
        eval_df.insert(0, "gold", "NA")
    else:
        eval_df["gold"] = (
            eval_df["gold"].fillna("").astype(str).str.strip().str.upper()
        )

    if EVAL_ROW_LIMIT is not None:
        eval_df = eval_df.head(EVAL_ROW_LIMIT)

    responses = generate_gpt_response_with_relations(prompt_builder, eval_df)
    parsed_predictions = parse_multiple_responses(responses)

    results_df = pd.DataFrame(
        {
            "gold": eval_df["gold"].fillna(""),
            "text": eval_df["text"].fillna(""),
        }
    )
    results_df["model_prediction_binary"] = parsed_predictions[
        "model_prediction"
    ].fillna("")
    results_df["model_prediction"] = parsed_predictions[
        "model_relation_label"
    ].fillna("")

    results_df.to_csv(
        RESULTS_CSV,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
    )
    print(f"Predictions saved to {RESULTS_CSV}")

    if few_shot_logs:
        log_df = pd.concat(few_shot_logs, ignore_index=True)
        log_df = log_df[
            ["input_text", "_source", "gold", "text"]
        ] if "_source" in log_df.columns else log_df
        log_df.to_csv(
            FEW_SHOT_LOG,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
        )
        print(f"Few-shot usage logged to {FEW_SHOT_LOG}")


if __name__ == "__main__":
    main()
