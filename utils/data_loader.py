from typing import Optional, Tuple

import pandas as pd


def get_dataframe(
    file_path: str,
    columns: Optional[Tuple[str, ...]] = ("gold", "text"),
    has_header: bool = True,
    *,
    keep_default_na: bool = False,
) -> pd.DataFrame:
    header = 0 if has_header else None
                                                                             
    df = pd.read_csv(file_path, header=header, keep_default_na=keep_default_na)

    if columns is not None:
        if not has_header:
            if len(df.columns) < len(columns):
                raise ValueError(
                    f"{file_path} has {len(df.columns)} columns, expected at least {len(columns)}."
                )
            new_columns = list(columns) + df.columns[len(columns):].tolist()
            df.columns = new_columns
        else:
            missing = [col for col in columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing expected columns {missing} in {file_path}.")
        df = df.loc[:, columns]

    return df
