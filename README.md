# Relation Extraction for Biodiversity Texts

This project extracts semantic relations between annotated biodiversity entities (e.g., species, environments, processes) using prompt-engineered large language models. It combines static few-shot demonstrations with LangChain-powered dynamic example selection to keep the prompts relevant for each sentence that needs to be classified.

## Table of Contents

- [Overview](#overview)
- [LangChain Architecture](#langchain-architecture)
- [Prompting Strategy](#prompting-strategy)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
- [Evaluation and Logging](#evaluation-and-logging)
- [Data](#data)

## Overview

The pipeline classifies each input sentence into one of the supported relation labels (`HAVE`, `OCCUR_IN`, `INFLUENCE`) or `NA` whenever no explicit relation exists. Sentences already contain entity placeholders such as `@ORGANISM$` or `@ENVIRONMENT$`, so the model focuses on judging whether there is a relationship and which type it matches.

High-level flow:

1. Build or refresh few-shot examples from `data/train.csv`.
2. Assemble an instruction + few-shot prompt per sentence, optionally including a structured "code prompt".
3. Call the LLM to predict the binary relation flag and the relation label.
4. Persist predictions to `results.csv`.
5. Evaluate performance with `evaluation.py` using both binary and multi-class metrics.

## LangChain Architecture

LangChain is only required for optional dynamic few-shot selection. The core components live in `utils/langchain_shot_selector.py`:

- **CSV ingestion**: `CSVLoader` (when available) or a pandas fallback loads `gold`/`text` pairs from `data/train.csv`.
- **BalancedEntityPairSelector**: Extends LangChain's `BaseExampleSelector` to return entity-pair specific samples. It groups examples by the first two entity tags found in the text, then draws a balanced set of positive vs. `NA` samples while respecting a global budget.
- **LabelBalancedExampleSelector / EntityPairExampleSelector**: Lightweight selectors used as fallbacks when only per-label balancing is required.
- **Keyword biasing**: When several labels compete for an entity pair, simple keyword heuristics bias the sampling order to reflect the query context.

`main.py` wires everything together through `build_balanced_entity_pair_selector`. If LangChain is not installed, the code gracefully falls back to pandas-based selectors so the pipeline still runs.

## Prompting Strategy

All prompt construction logic lives in `utils/prompt_generator.py` and `main.py`:

- **Instruction block**: A concise task description reminding the model about entity annotations, reasoning steps, and the exact output format (`"1, RELATION"` or `"0, NA"`). It stresses that co-occurrence alone is insufficient.
- **Few-shot assembly**: `build_prompt_builder` merges several sources of examples:
  - Static base shots from `data/shot.csv` plus a handful of contrastive sentences hard-coded in `main.py`.
  - LangChain selectors that inject entity-pair matched samples and balanced positive/negative evidence per query.
  - Optional positive-only or NA-only selectors (helpers exist but are currently unused).
- **Code-style prompting**: When `code_prompt.txt` exists, `build_code_prompt_builder` injects the current sentence into a code template and appends few-shot snippets formatted as pseudo-code assignments (with `results = [1, Have]` etc.). This style encourages deterministic reasoning and is tracked as `prompt_style="code"` in the evaluation log.
- **Logging**: Every assembled few-shot frame can be recorded through `record_few_shot_examples`, producing `few_shot_log.csv` for later auditing.

## Project Structure

- `main.py` - orchestrates the end-to-end inference workflow described above.
- `evaluation.py` - computes binary/multi-class metrics, appends them to `evaluation_log.csv`, and captures how many dynamic few-shot examples/inputs were used.
- `utils/` - helper modules (`data_loader`, prompt generation, LangChain selectors, GPT helpers, etc.).
- `data/` - CSV inputs such as `train.csv`, `shot.csv`, `check.csv` (evaluation set).
- `prompts.txt` - snapshot of the latest prompt preview generated from the static few-shot pool.
- `few_shot_log.csv` - optional log of the exact examples injected per evaluated sentence.
- `results.csv` - model predictions for the current evaluation batch.
- `code_prompt.txt` - optional template enabling the structured "code" prompting style.
- `requirements.txt` - dependency list (LangChain is optional but recommended for richer selectors).

## Installation

```bash
git clone https://github.com/MilVosk/Thesis_.git
cd Thesis_
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

LangChain is only needed for dynamic few-shot selection. If you want that behavior, keep it in `requirements.txt` or install `langchain` + `langchain-community` manually.

## Running the Pipeline

1. Place the evaluation sentences in `data/check.csv` (header must include at least `text`; `gold` is optional but needed for metrics).
2. Optionally edit `code_prompt.txt` to switch between natural language and code-style prompting.
3. Run:

```bash
python main.py
```

Key artifacts:

- `results.csv` - contains `gold`, `text`, `model_prediction_binary`, `model_prediction`.
- `prompts.txt` - updated preview of the base prompt.
- `few_shot_log.csv` - lists the exact examples chosen for each evaluated input (if logging enabled).

## Evaluation and Logging

`evaluation.py` loads `results.csv`, computes:

- **Binary metrics**: Treats any non-empty `gold` as a positive relation and compares against `model_prediction_binary`.
- **Multi-class metrics**: Restricts to rows with annotated relations and compares `model_prediction` vs. `gold` using micro F1 and Hamming loss.

### About Hamming loss

Hamming loss measures the fraction of positions where the prediction disagrees with the reference label. For the binary task it quantifies how often the model flips "relation vs. no relation"; for the multi-class task it captures the proportion of misclassified relation types among the sentences that actually contain a relation. Lower values indicate better performance.

After printing scores, it appends a row to `evaluation_log.csv` that captures:

- Timestamp and source file used for scoring.
- Count of dynamic few-shot examples and unique inputs that triggered them (derived from `few_shot_log.csv`).
- Binary + multi-class metrics (4 decimal precision).
- Prompt style (`code` vs. `natural`).

To evaluate:

```bash
python evaluation.py
```

## Data

The repository does not ship datasets. Expected files inside `data/`:

- `train.csv` - raw labeled data with `gold,text` columns (no header by default).
- `shot.csv` - generated by `ensure_shot_examples` to store balanced few-shot samples per label.
- `check.csv` - evaluation/inference inputs referenced by `EVAL_CSV_PATH`.

Entity mentions inside each sentence must already be tagged with `@ENTITY_TYPE$` tokens so the model can reason about relationships without additional NER steps.
