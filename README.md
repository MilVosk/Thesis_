# Relation Extraction for Biodiversity Texts

This project extracts semantic relations between annotated biodiversity entities (e.g., species, environments, processes) using prompt-engineered large language models. It builds natural-language or code-style prompts and can add dynamically selected few-shot examples for each sentence.

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

1. Load the evaluation CSV and, in few-shot mode, the configured training CSV.
2. Assemble an instruction prompt per sentence, optionally with dynamically selected few-shot examples or a structured code-style prompt.
3. Call the LLM to predict the binary relation flag and the relation label.
4. Persist predictions to `artifacts/results/results.csv`.
5. Evaluate performance with `evaluation.py` using both binary and multi-class metrics.

## LangChain Architecture

Dynamic few-shot selection lives in `utils/langchain_shot_selector.py`:

- **CSV ingestion**: Few-shot training examples are loaded from the configured training CSV. Each training row should provide a `gold` relation label (`HAVE`, `OCCUR_IN`, `INFLUENCE`, or `NA`) and a `text` sentence. The loader uses LangChain's `CSVLoader` when available and falls back to `pandas.read_csv()` otherwise. The current default training file is `data/train.csv`; use `--train-csv data/german_train.csv` for the German training set.
- **BalancedEntityPairSelector**: Extends LangChain's `BaseExampleSelector` to return entity-pair specific samples. It groups examples by the first two entity tags found in the text, then draws a balanced set of positive vs. `NA` samples while respecting a global budget.
- **SemanticSimilarityExampleSelector (optional)**: When enabled it embeds every training sentence (OpenAI embeddings by default) and uses FAISS to retrieve the `k` closest examples per query, complementing the entity-pair signal.
- **LabelBalancedExampleSelector / EntityPairExampleSelector**: Lightweight helper selectors available for non-default selection flows.
- **Keyword biasing**: When several labels compete for an entity pair, simple keyword heuristics bias the sampling order to reflect the query context.

`main.py` wires everything together through `build_balanced_entity_pair_selector` and, when enabled, `build_semantic_similarity_selector`. If LangChain CSV loading is unavailable, the balanced selector can still load examples with pandas. If semantic-similarity dependencies are missing, the pipeline continues without semantic examples.

## Prompting Strategy

All prompt construction logic lives in `utils/prompt_generator.py` and `main.py`:

 - **Instruction block**: A concise task description reminding the model about entity annotations, reasoning steps, and the exact output format (`"1, RELATION"` or `"0, NA"`). It stresses that co-occurrence alone is insufficient.
 - **Few-shot assembly**: When `USE_ZERO_SHOT = False`, `build_prompt_builder` merges examples selected for the current sentence:
   - The balanced entity-pair selector draws positive and `NA` examples around matching entity pairs.
   - The semantic-similarity selector, when enabled, adds nearest training examples from a FAISS-backed vector store.
   - Optional positive-only and `NA`-only selector hooks exist in the code but are not used by the default pipeline.
 - **Output constraints**: The natural-language prompt spells out the only valid responses (`1, HAVE|OCCUR_IN|INFLUENCE` or `0, NA`) so the LLM cannot drift into prose answers.
 - **Deterministic calls**: `utils/gpt_utils.py` escapes each input sentence and calls `gpt-4o-mini` with `temperature=0`/bounded `max_tokens` for reproducible classifications.
 - **Code-style prompting**: When `USE_CODE_PROMPT = True` in `main.py`, the pipeline uses `prompts/code_prompts.txt` instead of the natural-language prompt. In few-shot mode, `build_code_prompt_builder` appends selected examples as pseudo-code assignments (with `results = [1, Have]` etc.). This style is tracked as `prompt_style="code"` in the evaluation log.
 - **Logging**: In few-shot mode, every assembled few-shot frame can be recorded through `record_few_shot_examples`, producing `artifacts/logs/few_shot_log.csv` for later auditing.

The pipeline currently defaults to zero-shot mode (`USE_ZERO_SHOT = True`). If few-shot mode is enabled, the current configuration uses up to one positive entity-pair example, two `NA` entity-pair examples, and two semantic-similarity examples per input, subject to the selector limits in `main.py` and `utils/langchain_shot_selector.py`.

## Project Structure

- `main.py` - orchestrates the end-to-end inference workflow described above.
- `evaluation.py` - computes binary and multi-class F1, appends results to `artifacts/metrics/evaluation_log.csv`, and captures how many dynamic few-shot examples/inputs were used.
- `utils/` - helper modules (`data_loader`, prompt generation, LangChain selectors, GPT helpers, etc.).
- `data/` - CSV inputs such as `german_train.csv`, `german_test.csv`, `train.csv`, and `test.csv`.
- `prompts/` - houses the managed prompt assets (`natural_language_prompt.txt`, `natural_language_prompt_de.txt`, `code_prompts.txt`) plus the zero-shot `prompt_preview.txt`.
- `artifacts/` - run outputs separated into `results/` (e.g., `results/results.csv`), `logs` (`logs/few_shot_log.csv`), and `metrics` (`metrics/evaluation_log.csv`, `metrics/evaluation_summary.csv`).
- `requirements.txt` - dependency list for the default OpenAI, LangChain, and FAISS-based pipeline.

## Installation

```bash
git clone https://github.com/MilVosk/Thesis_.git
cd Thesis_
python -m venv .venv
.\\.venv\\Scripts\\activate
pip install -r requirements.txt
```

On macOS/Linux/WSL, activate the environment with:

```bash
source .venv/bin/activate
```

LangChain, FAISS, and `langchain-openai` are needed for the default semantic few-shot selector. They are included in `requirements.txt`. To run without semantic retrieval, set `USE_SEMANTIC_SELECTOR = False` in `main.py`.

## Running the Pipeline

1. Set `OPENAI_API_KEY` in your environment or in a local `.env` file.
2. Place the evaluation sentences in the configured evaluation CSV. The current default is `data/test.csv`; the file must include at least a `text` column, and `gold` is optional but needed for metrics.
3. Run with the default English natural-language prompt:

```bash
python main.py
```

4. To use the German natural-language prompt (`prompts/natural_language_prompt_de.txt`), run:

```bash
python main.py --natural-prompt-lang de
```

5. To override the input files, pass explicit CSV paths:

```bash
python main.py --eval-csv data/german_test.csv --train-csv data/german_train.csv --natural-prompt-lang de
```

If you want German data to be the default every time, update the defaults near the top of `main.py`:

```python
EVAL_CSV_PATH_DEFAULT = "data/german_test.csv"
TRAIN_CSV_PATH_DEFAULT = "data/german_train.csv"
```

Natural-language prompting is used when `USE_CODE_PROMPT = False` in `main.py`. If `USE_CODE_PROMPT = True`, the pipeline uses `prompts/code_prompts.txt` instead of the natural-language prompt files.

Key artifacts:

- `artifacts/results/results.csv` - contains `gold`, `text`, `model_prediction_binary`, `model_prediction`.
- `prompts/prompt_preview.txt` - preview of the zero-shot base prompt, written during zero-shot runs.
- `artifacts/logs/few_shot_log.csv` - lists the exact examples chosen for each evaluated input (if logging enabled).

Embedding-driven retrieval is controlled by `USE_SEMANTIC_SELECTOR` and `SEMANTIC_SIMILARITY_SAMPLES` in `main.py`. The helper defaults to `OpenAIEmbeddings` and FAISS, so semantic retrieval requires the LangChain/OpenAI/FAISS packages from `requirements.txt` and a configured `OPENAI_API_KEY`.

## Evaluation and Logging

`evaluation.py` loads `artifacts/results/results.csv`, computes:

- **Binary F1**: Treats `gold` rows that are neither empty nor `"NA"` as positives (true relations) before comparing against `model_prediction_binary`.
- **Multi-class micro F1**: Restricts to rows with annotated relations and compares `model_prediction` vs. `gold`.

After printing scores, it appends rows to `artifacts/metrics/evaluation_log.csv` and `artifacts/metrics/evaluation_summary.csv`. These logs capture:

- Timestamp and source file used for scoring.
- Count and averages for dynamic few-shot examples where available (derived from `artifacts/logs/few_shot_log.csv`).
- Binary and multi-class F1 scores (4 decimal precision).
- Prompt style (`code` vs. `natural`).

When at least three matching runs are available, `evaluation.py` also appends their average metrics to `artifacts/metrics/evaluation_average.csv`.

To evaluate:

```bash
python evaluation.py
```

## Data

Expected files inside `data/`:

- `train.csv` - default few-shot training pool, read as no-header CSV with `gold,text` columns.
- `test.csv` - default evaluation/inference file, read with a header row and at least a `text` column.
- `german_train.csv` / `german_test.csv` - German datasets that can be selected with `--train-csv`, `--eval-csv`, and `--natural-prompt-lang de`.

Entity mentions inside each sentence must already be tagged with `@ENTITY_TYPE$` tokens so the model can reason about relationships without additional NER steps.
