import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.metrics import f1_score, hamming_loss

from utils.data_loader import get_dataframe

try:
    from main import USE_ZERO_SHOT
except ImportError:  # fallback when main isn't importable
    USE_ZERO_SHOT = False


def load_results(path: str = "results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gold"] = df["gold"].fillna("").str.strip()
    df["model_prediction"] = df["model_prediction"].fillna("").str.strip()
    return df


def compute_binary_metrics(df: pd.DataFrame) -> Tuple[float, float]:
    """Return F1 and Hamming loss for relation vs no-relation."""
    normalized_gold = df["gold"].fillna("").str.strip().str.upper()
    binary_gold = (normalized_gold != "") & (normalized_gold != "NA")
    binary_gold = binary_gold.astype(int)
    binary_pred = df["model_prediction_binary"].astype(int)
    f1 = f1_score(binary_gold, binary_pred, average="binary")
    h_loss = hamming_loss(binary_gold, binary_pred)
    return f1, h_loss


def compute_multiclass_metrics(
    df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float]]:
    """Return F1 and Hamming loss for multi-class prediction or (None, None)."""
    mask_has_relation = df["gold"] != ""
    multi_df = df[mask_has_relation].copy()
    if multi_df.empty:
        return None, None

    multi_df["gold"] = multi_df["gold"].str.upper()
    multi_df["model_prediction"] = multi_df["model_prediction"].str.upper()
    f1 = f1_score(multi_df["gold"], multi_df["model_prediction"], average="micro")
    h_loss = hamming_loss(multi_df["gold"], multi_df["model_prediction"])
    return f1, h_loss


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
    df = load_results()

    binary_f1, binary_hamming = compute_binary_metrics(df)
    print(f"Binary F1 (relation vs none): {binary_f1:.2f}")
    print(f"Binary Hamming loss: {binary_hamming:.2f}")

    multi_f1, multi_hamming = compute_multiclass_metrics(df)
    if multi_f1 is None:
        print("No gold relations available for multi-class evaluation.")
    else:
        print(f"Multi-class micro F1: {multi_f1:.2f}")
        print(f"Multi-class Hamming loss: {multi_hamming:.2f}")

    log_path = Path("evaluation_log.csv")
    few_shot_log = Path("few_shot_log.csv")
    zero_shot_mode = bool(USE_ZERO_SHOT)
    few_shot_examples = "0"
    few_shot_inputs = "0"
    # Number of few-shot examples shown to the model per input (0 for pure zero-shot).
    examples_per_input = "0"
    avg_positive_per_input = "0"
    avg_negative_per_input = "0"
    avg_semantic_per_input = "0"
    prompt_style = (
        "code"
        if Path("code_prompt.txt").exists()
        and Path("code_prompt.txt").read_text(encoding="utf-8").strip()
        else "natural"
    )
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
        "binary_hamming_loss",
        "multi_class_f1",
        "multi_class_hamming_loss",
        "prompt_style",
    ]
    timestamp = datetime.utcnow().isoformat()
    log_row = {
        "timestamp": timestamp,
        "results_source": "results.csv",
        "few_shot_examples": few_shot_examples,
        "few_shot_inputs": few_shot_inputs,
        "examples_per_input": examples_per_input,
        "binary_f1": f"{binary_f1:.4f}",
        "binary_hamming_loss": f"{binary_hamming:.4f}",
        "multi_class_f1": "" if multi_f1 is None else f"{multi_f1:.4f}",
        "multi_class_hamming_loss": ""
        if multi_hamming is None
        else f"{multi_hamming:.4f}",
        "prompt_style": prompt_style,
    }
    append_csv_row(log_path, fieldnames, log_row)

    detailed_log_path = Path("evaluation_summary.csv")
    detailed_fields = [
        "timestamp",
        "train_instances",
        "test_instances",
        "positive_per_input",
        "negative_per_input",
        "semantic_per_input",
        "binary_f1",
        "binary_hamming_loss",
        "multi_class_f1",
        "multi_class_hamming_loss",
        "prompt_style",
    ]
    train_instances = count_training_instances()
    detailed_row = {
        "timestamp": timestamp,
        "train_instances": "" if train_instances is None else str(train_instances),
        "test_instances": str(len(df)),
        "positive_per_input": avg_positive_per_input,
        "negative_per_input": avg_negative_per_input,
        "semantic_per_input": avg_semantic_per_input,
        "binary_f1": f"{binary_f1:.4f}",
        "binary_hamming_loss": f"{binary_hamming:.4f}",
        "multi_class_f1": "" if multi_f1 is None else f"{multi_f1:.4f}",
        "multi_class_hamming_loss": ""
        if multi_hamming is None
        else f"{multi_hamming:.4f}",
        "prompt_style": prompt_style,
    }
    append_csv_row(detailed_log_path, detailed_fields, detailed_row)


if __name__ == "__main__":
    main()
