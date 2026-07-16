from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

                    
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = ARTIFACTS_DIR / "logs"
RESULTS_DIR = ARTIFACTS_DIR / "results"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PROMPTS_DIR = BASE_DIR / "prompts"

                             
RESULTS_CSV = RESULTS_DIR / "results.csv"
CODE_PROMPTS_FILE = PROMPTS_DIR / "code_prompts.txt"
NATURAL_PROMPT_FILE = PROMPTS_DIR / "natural_language_prompt.txt"
NATURAL_PROMPT_DE_FILE = PROMPTS_DIR / "natural_language_prompt_de.txt"
PROMPT_PREVIEW_FILE = PROMPTS_DIR / "prompt_preview.txt"
FEW_SHOT_LOG = LOGS_DIR / "few_shot_log.csv"
EVAL_LOG = METRICS_DIR / "evaluation_log.csv"
EVAL_SUMMARY = METRICS_DIR / "evaluation_summary.csv"
EVAL_AVERAGE = METRICS_DIR / "evaluation_average.csv"


def ensure_directories() -> None:
    """Create expected directories if they do not exist yet."""
    for path in (
        ARTIFACTS_DIR,
        LOGS_DIR,
        RESULTS_DIR,
        METRICS_DIR,
        PROMPTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
