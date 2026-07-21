import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score

from paths import (
    EVAL_LOG,
    FEW_SHOT_LOG,
    RESULTS_CSV,
    ensure_directories,
)

try:
    from main import USE_ZERO_SHOT, USE_CODE_PROMPT
except ImportError:                                       
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

    examples_per_input = "0"
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

                if "input_text" in fs_df.columns:
                    per_input_counts = fs_df.groupby("input_text").size()
                    if not per_input_counts.empty:
                        avg_n = per_input_counts.mean()
                        examples_per_input = str(max(0, int(round(avg_n))))
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


if __name__ == "__main__":
    main()
