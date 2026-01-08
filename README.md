# Biodiversity Relation Extraction

This repository implements a prompt-engineered pipeline for extracting semantic relations between annotated biodiversity entities. It combines static few-shot prompts, dynamic entity-pair sampling, and evaluation tooling to iteratively refine large language model (LLM) behaviour on curated datasets.

## Contents

- [System Overview](#system-overview)
- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
- [How Few-Shot Selection Works](#how-few-shot-selection-works)
- [Prompt Generation](#prompt-generation)
- [Running Inference](#running-inference)
- [Evaluation Workflow](#evaluation-workflow)
- [Adding New Data](#adding-new-data)
- [Troubleshooting & Tips](#troubleshooting--tips)

## System Overview

1. **Data ingestion**: Training and evaluation sentences reside in `data/train.csv`, `data/check.csv`, or `data/test.csv`. Each row contains annotated entities (e.g., `@ORGANISM$`, `@ENVIRONMENT$`) plus optional gold labels.
2. **Static few-shot creation**: We build `data/shot.csv` by sampling balanced examples per label (HAVE, OCCUR_IN, INFLUENCE, NA) using LangChain-based selectors.
3. **Dynamic few-shot augmentation**: For every inference sentence we select additional examples whose entity pairs match the input sentence. These augment the static prompt.
4. **Prompt assembly**: `utils/prompt_generator.py` merges the instruction block, relation heuristics, and all few-shot rows into a single prompt saved to `prompts.txt`.
5. **LLM inference**: `main.py` streams each sentence through the prompt, collects the model responses, parses them into structured predictions, and writes `results.csv`.
6. **Evaluation**: `evaluation.py` reads `results.csv`, computes binary and multi-class metrics (F1, Hamming loss), and logs them with metadata to `evaluation_log.csv`.

## Repository Layout

```
├── data/
│   ├── train.csv         # training rows used for few-shot selection
│   ├── check.csv         # small validation set
│   ├── test.csv          # large evaluation set
│   └── shot.csv          # static few-shot examples (generated)
├── utils/
│   ├── prompt_generator.py        # prompt assembly logic
│   ├── langchain_shot_selector.py # balanced selectors & entity-pair sampling
│   ├── extract_shots.py           # CLI for building shot.csv
│   ├── data_loader.py             # CSV helper with NA handling
│   └── gpt_utils.py               # OpenAI client + response parser
├── main.py               # end-to-end inference entry point
├── evaluation.py         # metrics and evaluation logging
├── prompts.txt           # latest prompt sent to the LLM (auto-generated)
├── results.csv           # latest model predictions
├── evaluation_log.csv    # history of evaluation runs
└── README.md             # this document
```

## Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MilVosk/Thesis_.git
   cd Thesis_
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set OpenAI credentials**
   - Copy `.env.example` to `.env` (if provided) or create `.env`.
   - Add `OPENAI_API_KEY=...` so `utils/gpt_utils.py` can authenticate.

## How Few-Shot Selection Works

### Static few-shot (shot.csv)

- Run `python utils/extract_shots.py` or let `main.py` call `ensure_shot_examples`.
- `utils/langchain_shot_selector.py` reads `data/train.csv` via LangChain’s `CSVLoader`.
- `LabelBalancedExampleSelector` shuffles and samples `samples_per_label` rows from each class.
- Output is written to `data/shot.csv`. Each row has `gold` and `text`.

### Dynamic entity-pair selections

- `build_balanced_entity_pair_selector` groups training rows by their first two entity tags.
- For each inference sentence, the selector:
  1. Extracts all entity pairs in the input text.
  2. For each pair (up to `max_pairs`), samples `positive_samples` non-NA and `na_samples` NA examples from matching pairs or fallbacks.
  3. Deduplicates and caps the total appended rows (`max_total_examples`).
- This ensures the prompt contains examples tailored to the sentence’s entity types, even if they weren’t in the static few-shot file.

## Prompt Generation

`utils/prompt_generator.py` builds the final prompt:

- **Instruction block**: States the task, entity schema, and two-step reasoning (INFLUENCE/OCCUR/HAVE). It emphasizes:
  - Analyze the sentence first.
  - Distinguish influence vs occurrence vs property relations.
  - Treat metadata/hypotheses as NA.
  - Never leave the final answer blank.
- **Relation mapping**: Lists which entity-type pairs typically map to HAVE, OCCUR_IN, or INFLUENCE.
- **Few-shot examples**: Appends every row from `data/shot.csv` plus the dynamic entity-pair samples for the current sentence.
- The prompt is written to `prompts.txt` so you can inspect what the model saw.

## Running Inference

```bash
python main.py
```

Steps performed:

1. Ensure `data/shot.csv` exists (create if necessary).
2. Load `data/check.csv` or `data/test.csv` (configurable in `main.py`).
3. Build the prompt builder with dynamic selectors.
4. Call OpenAI’s ChatCompletions for each sentence, passing the assembled prompt plus the sentence under classification.
5. Parse responses with `parse_multiple_responses`, producing:
   - `model_prediction_binary` (0/1)
   - `model_prediction` (NA or relation label)
6. Save results to `results.csv`.

## Evaluation Workflow

1. Run `python evaluation.py` after inference. It:
   - Reads `results.csv`.
   - Computes binary and multi-class F1 + Hamming loss.
   - Logs metrics, timestamp, and few-shot counts (from `few_shot_log.csv`) to `evaluation_log.csv`.
2. Inspect the log to compare runs. Example entry:
   ```
   timestamp,binary_f1,binary_hamming_loss,multi_class_f1, ...
   2025-12-11T13:57:02Z,0.667,0.38,0.762,0.238
   ```
3. Analyze failure cases by opening `results.csv` or `few_shot_log.csv`. NA false positives often stem from summary-like sentences; genuine relations missed as NA may look like structured metadata.

## Adding New Data

- Append rows to `data/train.csv` (entity tags + `gold` label) to enrich few-shot pools.
- For quick validation sets, add rows to `data/check.csv` (small curated subset).
- For large evaluations use `data/test.csv`. Configure `EVAL_CSV_PATH` in `main.py` if needed.
- Ensure new rows follow the same annotation convention (`@ENTITY_TYPE$` placeholders).

## Troubleshooting & Tips

- **Prompt inspection**: After running `main.py`, open `prompts.txt` to audit the exact instructions and few-shot examples the model saw.
- **Few-shot logs**: Check `few_shot_log.csv` to understand which static/dynamic examples were attached per sentence (`_source` column distinguishes base vs entity-pair vs NA selectors).
- **Binary vs Multi-class trade-offs**: Increasing `na_samples` or adding NA-focused instructions improves binary adherence but can lower multi-class accuracy by reducing positive context. Adjust `positive_samples`, `na_samples`, and `max_total_examples` to balance these metrics.
- **Blank predictions**: The prompt explicitly forbids blank answers; if the parser still sees empty labels, use `parse_multiple_responses` to coerce blanks to `0/NA`.
- **LangChain availability**: If LangChain isn’t installed, `utils/langchain_shot_selector.py` falls back to Pandas CSV loading, so static few-shot selection still works.
- **OpenAI rate limits**: `main.py` sends each sentence sequentially. For large datasets, consider batching or adding retry logic.

With these components you can iteratively refine LLM prompts, few-shot selection, and evaluation metrics for biodiversity relation extraction. Update the prompt or few-shot pools, rerun `main.py`, then regenerate metrics via `evaluation.py` to measure improvements.
