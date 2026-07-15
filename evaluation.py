import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score

from paths import (
    EVAL_LOG,
    EVAL_SUMMARY,
    EVAL_AVERAGE,
    FEW_SHOT_LOG,
    RESULTS_CSV,
    ensure_directories,
)
from utils.data_loader import get_dataframe

try:
    from main import USE_ZERO_SHOT, USE_CODE_PROMPT
except ImportError:  # fallback when main isn't importable
    USE_ZERO_SHOT = False
    USE_CODE_PROMPT = False


def load_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gold"] = df["gold"].fillna("").str.strip()
    df["model_prediction"] = df["model_prediction"].fillna("").str.strip()
    return df


def compute_binary_f1(df: pd.DataFrame) -> float:
    """Return F1 for relation vs no-relation."""
    normalized_gold = df["gold"].fillna("").str.strip().str.upper()
    binary_gold = (normalized_gold != "") & (normalized_gold != "NA")
    binary_gold = binary_gold.astype(int)
    binary_pred = df["model_prediction_binary"].astype(int)
    return f1_score(binary_gold, binary_pred, average="binary")


def compute_multiclass_f1(
    df: pd.DataFrame,
) -> Optional[float]:
    """Return micro F1 for multi-class prediction, or None if no gold relations exist."""
    normalized_gold = df["gold"].fillna("").str.strip().str.upper()
    mask_has_relation = (normalized_gold != "") & (normalized_gold != "NA")
    multi_df = df[mask_has_relation].copy()
    if multi_df.empty:
        return None

    multi_df["gold"] = normalized_gold[mask_has_relation]
    multi_df["model_prediction"] = (
        multi_df["model_prediction"].fillna("").astype(str).str.strip().str.upper()
    )
    return f1_score(multi_df["gold"], multi_df["model_prediction"], average="micro")


def count_training_instances(path: str = "data/train.csv") -> Optional[int]:
    csv_path = Path(path)
    if not csv_path.exists():
        return None

    for has_header in (False, True):
        try:
            df = get_dataframe(
                path,
                columns=("gold", "text"),
                has_header=has_header,
                keep_default_na=False,
            )
            return len(df)
        except Exception:
            continue

    try:
        df = pd.read_csv(path)
        return len(df)
    except Exception:
        return None


def append_csv_row(log_path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    def _write(df: pd.DataFrame) -> None:
        df.to_csv(
            log_path,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL,
        )

    if log_path.exists():
        try:
            existing = pd.read_csv(log_path, keep_default_na=False)
        except Exception:
            existing = None
        if existing is not None:
            for field in fieldnames:
                if field not in existing.columns:
                    existing[field] = ""
            ordered = existing[fieldnames]
            updated = pd.concat(
                [ordered, pd.DataFrame([row])],
                ignore_index=True,
            )
            _write(updated[fieldnames])
            return

    with log_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    def detect_language(text_series: pd.Series) -> str:
        """Lightweight heuristic detector (de vs en) without extra deps."""
        sample = " ".join(text_series.fillna("").astype(str).head(200).tolist()).lower()
        german_score = 0
        english_score = 0

        # Umlauts / ß strongly indicate German.
        german_score += len(re.findall(r"[äöüß]", sample))

        german_markers = [
            " der ", " die ", " und ", " mit ", " für ", " auf ", " nicht ",
            " zum ", " im ", " des ", " von ", " ist ", " eine ", " einen ", " einem "
        ]
        english_markers = [
            " the ", " and ", " of ", " in ", " for ", " with ", " is ", " are ",
            " to ", " from ", " that ", " this ", " an "
        ]

        german_score += sum(sample.count(tok) for tok in german_markers)
        english_score += sum(sample.count(tok) for tok in english_markers)

        return "de" if german_score > english_score else "en"

    ensure_directories()
    df = load_results()

    binary_f1 = compute_binary_f1(df)
    print(f"Binary F1 (relation vs none): {binary_f1:.2f}")

    multi_f1 = compute_multiclass_f1(df)
    if multi_f1 is None:
        print("No gold relations available for multi-class evaluation.")
    else:
        print(f"Multi-class micro F1: {multi_f1:.2f}")

    log_path = EVAL_LOG
    few_shot_log = FEW_SHOT_LOG
    zero_shot_mode = bool(USE_ZERO_SHOT)
    few_shot_examples = "0"
    few_shot_inputs = "0"
    # Number of few-shot examples shown to the model per input (0 for pure zero-shot).
    examples_per_input = "0"
    avg_positive_per_input = "0"
    avg_negative_per_input = "0"
    avg_semantic_per_input = "0"
    prompt_style = "code" if USE_CODE_PROMPT else "natural"
    if not zero_shot_mode and few_shot_log.exists():
        try:
            fs_df = pd.read_csv(few_shot_log, keep_default_na=False)
            if not fs_df.empty:
                if "_source" not in fs_df.columns:
                    fs_df["_source"] = ""
                fs_df["_source"] = fs_df["_source"].astype(str)
                dynamic_df = fs_df[fs_df["_source"] != "base"]
                few_shot_examples = str(len(dynamic_df))
                if "input_text" in dynamic_df.columns:
                    few_shot_inputs = str(dynamic_df["input_text"].nunique())
                # Derive average number of examples shown per input.
                if "input_text" in fs_df.columns:
                    per_input_counts = fs_df.groupby("input_text").size()
                    if not per_input_counts.empty:
                        avg_n = per_input_counts.mean()
                        examples_per_input = str(max(0, int(round(avg_n))))

                if "input_text" in dynamic_df.columns:
                    entity_df = dynamic_df[
                        dynamic_df["_source"].str.lower() != "semantic"
                    ].copy()
                    if not entity_df.empty:
                        entity_df["gold_norm"] = (
                            entity_df["gold"].astype(str).str.strip().str.upper()
                        )
                        positive_counts = (
                            entity_df[entity_df["gold_norm"] != "NA"]
                            .groupby("input_text")
                            .size()
                        )
                        if not positive_counts.empty:
                            avg_positive_per_input = f"{positive_counts.mean():.2f}"
                        negative_counts = (
                            entity_df[entity_df["gold_norm"] == "NA"]
                            .groupby("input_text")
                            .size()
                        )
                        if not negative_counts.empty:
                            avg_negative_per_input = f"{negative_counts.mean():.2f}"

                    semantic_df = dynamic_df[
                        dynamic_df["_source"].str.lower() == "semantic"
                    ]
                    if not semantic_df.empty:
                        semantic_counts = semantic_df.groupby("input_text").size()
                        if not semantic_counts.empty:
                            avg_semantic_per_input = f"{semantic_counts.mean():.2f}"
        except Exception:
            pass

    fieldnames = [
        "timestamp",
        "results_source",
        "few_shot_examples",
        "few_shot_inputs",
        "examples_per_input",
        "binary_f1",
        "multi_class_f1",
        "prompt_style",
    ]
    timestamp = datetime.utcnow().isoformat()
    def _int_str(val: str) -> str:
        try:
            return str(int(round(float(val))))
        except Exception:
            return val

    log_row = {
        "timestamp": timestamp,
        "results_source": RESULTS_CSV.name,
        "few_shot_examples": few_shot_examples,
        "few_shot_inputs": few_shot_inputs,
        "examples_per_input": _int_str(examples_per_input),
        "binary_f1": f"{binary_f1:.4f}",
        "multi_class_f1": "" if multi_f1 is None else f"{multi_f1:.4f}",
        "prompt_style": prompt_style,
    }
    append_csv_row(log_path, fieldnames, log_row)

    detailed_log_path = EVAL_SUMMARY
    detailed_fields = [
        "timestamp",
        "train_instances",
        "test_instances",
        "test_language",
        "positive_per_input",
        "negative_per_input",
        "semantic_per_input",
        "binary_f1",
        "multi_class_f1",
        "prompt_style",
    ]
    train_instances = count_training_instances()
    detected_language = detect_language(df["text"]) if "text" in df.columns else "en"
    detailed_row = {
        "timestamp": timestamp,
        "train_instances": "" if train_instances is None else str(train_instances),
        "test_instances": str(len(df)),
        "test_language": detected_language,
        "positive_per_input": _int_str(avg_positive_per_input),
        "negative_per_input": _int_str(avg_negative_per_input),
        "semantic_per_input": _int_str(avg_semantic_per_input),
        "binary_f1": f"{binary_f1:.4f}",
        "multi_class_f1": "" if multi_f1 is None else f"{multi_f1:.4f}",
        "prompt_style": prompt_style,
    }
    append_csv_row(detailed_log_path, detailed_fields, detailed_row)

    # If we have at least 3 runs for the same combination, log their average.
    try:
        if EVAL_SUMMARY.exists():
            summary_df = pd.read_csv(EVAL_SUMMARY, keep_default_na=False)
            combo_mask = (
                (summary_df["prompt_style"] == detailed_row["prompt_style"])
                & (summary_df["test_language"] == detailed_row["test_language"])
                & (summary_df["positive_per_input"] == detailed_row["positive_per_input"])
                & (summary_df["negative_per_input"] == detailed_row["negative_per_input"])
                & (summary_df["semantic_per_input"] == detailed_row["semantic_per_input"])
            )
            combo_df = summary_df.loc[combo_mask].copy()
            if len(combo_df) >= 3:
                combo_df = combo_df.sort_values("timestamp").tail(3)
                def _mean(col: str) -> Optional[float]:
                    vals = pd.to_numeric(combo_df[col], errors="coerce").dropna()
                    return None if vals.empty else vals.mean()

                avg_binary_f1 = _mean("binary_f1")
                avg_multi_f1 = _mean("multi_class_f1")

                avg_fields = [
                    "timestamp",
                    "runs_averaged",
                    "prompt_style",
                    "test_language",
                    "positive_per_input",
                    "negative_per_input",
                    "semantic_per_input",
                    "binary_f1",
                    "multi_class_f1",
                ]
                avg_row = {
                    "timestamp": timestamp,
                    "runs_averaged": "3",
                    "prompt_style": detailed_row["prompt_style"],
                    "test_language": detailed_row["test_language"],
                    "positive_per_input": detailed_row["positive_per_input"],
                    "negative_per_input": detailed_row["negative_per_input"],
                    "semantic_per_input": detailed_row["semantic_per_input"],
                    "binary_f1": "" if avg_binary_f1 is None else f"{avg_binary_f1:.4f}",
                    "multi_class_f1": "" if avg_multi_f1 is None else f"{avg_multi_f1:.4f}",
                }
                append_csv_row(EVAL_AVERAGE, avg_fields, avg_row)
    except Exception:
        pass


if __name__ == "__main__":
    main()
