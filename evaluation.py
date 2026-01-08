import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.metrics import f1_score, hamming_loss


def load_results(path: str = "results.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gold"] = df["gold"].fillna("").str.strip()
    df["model_prediction"] = df["model_prediction"].fillna("").str.strip()
    return df


def compute_binary_metrics(df: pd.DataFrame) -> Tuple[float, float]:
    """Return F1 and Hamming loss for relation vs no-relation."""
    binary_gold = (df["gold"] != "").astype(int)
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
    few_shot_examples = ""
    few_shot_inputs = ""
    prompt_style = "code" if Path("code_prompt.txt").exists() and Path("code_prompt.txt").read_text(encoding="utf-8").strip() else "natural"
    if few_shot_log.exists():
        try:
            fs_df = pd.read_csv(few_shot_log, keep_default_na=False)
            dynamic_mask = fs_df.get("_source", "").astype(str) != "base"
            dynamic_df = fs_df[dynamic_mask]
            few_shot_examples = str(len(dynamic_df))
            if "input_text" in dynamic_df.columns:
                few_shot_inputs = str(dynamic_df["input_text"].nunique())
        except Exception:
            pass

    fieldnames = [
        "timestamp",
        "results_source",
        "few_shot_examples",
        "few_shot_inputs",
        "binary_f1",
        "binary_hamming_loss",
        "multi_class_f1",
        "multi_class_hamming_loss",
        "prompt_style",
    ]
    log_row = {
        "timestamp": datetime.utcnow().isoformat(),
        "results_source": "results.csv",
        "few_shot_examples": few_shot_examples,
        "few_shot_inputs": few_shot_inputs,
        "binary_f1": f"{binary_f1:.4f}",
        "binary_hamming_loss": f"{binary_hamming:.4f}",
        "multi_class_f1": "" if multi_f1 is None else f"{multi_f1:.4f}",
        "multi_class_hamming_loss": ""
        if multi_hamming is None
        else f"{multi_hamming:.4f}",
        "prompt_style": prompt_style,
    }
    def _write_log(rows: pd.DataFrame) -> None:
        rows.to_csv(
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
                [ordered, pd.DataFrame([log_row])],
                ignore_index=True,
            )
            _write_log(updated[fieldnames])
            return

    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(log_row)


if __name__ == "__main__":
    main()
