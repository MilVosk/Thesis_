from __future__ import annotations

from pathlib import Path

from utils.langchain_shot_selector import select_few_shot_examples


def extract_shots(
    source_csv: str | Path = "data/train.csv",
    target_csv: str | Path = "data/shot.csv",
    label_column: str = "gold",
    text_column: str = "text",
    samples_per_label: int = 20,
    has_header: bool = False,
) -> None:

    selected_df = select_few_shot_examples(
        source_csv=source_csv,
        label_column=label_column,
        text_column=text_column,
        samples_per_label=samples_per_label,
        has_header=has_header,
    )

    shots_path = Path(target_csv)
    shots_path.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(shots_path, index=False)


if __name__ == "__main__":
    extract_shots()
