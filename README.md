# Leveraging Large Language Models for Biodiversity Relation Extraction: A Code vs. Natural Prompting Study in English and German

This project explores instruction-based prompting for biodiversity relation extraction in English and German. It compares zero-shot and few-shot settings for classifying relations between already annotated entity mentions in biodiversity texts. The experiments test two prompting styles, code-style prompting and natural-language prompting, on both English and German language data.

## Overview

The task is sentence-level relation extraction. Each input sentence already contains marked entity mentions, such as `@ORGANISM$`, `@ENVIRONMENT$`, etc. Given a sentence with two marked entities, the model predicts whether an explicit relation exists and which relation type applies.

The supported labels are:

- `HAVE`
- `OCCUR_IN`
- `INFLUENCE`
- `NA`

`NA` is used when no explicit relation is expressed between the entity pair. 

## Dataset

The experiments are based on the BiodivRE [Link text](https://github.com/fusion-jena/BiodivNERE) relation extraction dataset. The original dataset is in English, and a German version was created through machine translation for the cross-lingual experiments.

The original train/test split is kept:

- The training set is used only for few-shot example retrieval.
- The test set is used for final inference and evaluation.

Expected files inside `data/`:

- `train.csv` - English training examples for few-shot retrieval.
- `test.csv` - English test examples for evaluation.
- `german_train.csv` - German training examples for few-shot retrieval.
- `german_test.csv` - German test examples for evaluation.
- `dev.csv` - development data, if needed for analysis.


## Prompting Setup

The repository supports two prompt styles, and both are tested on English and German data:

- Natural-language (NL-style) prompts, stored in `prompts/natural_language_prompt.txt` and `prompts/natural_language_prompt_de.txt`.
- Code-style prompts, stored in `prompts/code_prompts.txt`.

Both prompt styles can be used in zero-shot and few-shot settings. In zero-shot mode, the model receives only the task instructions and the input sentence. In few-shot mode, the prompt is extended with dynamically selected examples from the training set.

Prompt components include:

- Task definition.
- Entity-type constraints.
- Relation-specific guidance.
- Checklist-style instructions.
- Strict output format: `1, RELATION` or `0, NA`.

Few-shot retrieval is implemented in `utils/langchain_shot_selector.py`. The default setup uses entity-pair balanced selection and can optionally add semantic-similarity examples through LangChain, OpenAI embeddings, and FAISS.

## Repository Structure

```text
.
├── data/                  # Train, test, and development CSV files
├── prompts/               # Natural-language and code-style prompt templates
├── utils/                 # Data loading, prompt generation, LLM calls, example selection
├── artifacts/             # Generated run outputs, ignored by Git
│   ├── results/           # Prediction CSV files
│   ├── metrics/           # Evaluation logs and metric summaries
│   ├── logs/              # Few-shot example logs
├── main.py                # Runs relation extraction inference
├── evaluation.py          # Computes binary and multi-class evaluation metrics
├── paths.py               # Shared paths for generated outputs
├── translator.py          # Translation helper used for German data preparation
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/MilVosk/Thesis_.git
cd Thesis_
python -m venv .venv
```

On Windows:

```bash
.\.venv\Scripts\activate
```

On macOS, Linux, or WSL:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your OpenAI API key in the environment or in a local `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Model Configuration

The default OpenAI model is `gpt-4o-mini`.

For relation extraction experiments, change the `model` value in `utils/gpt_utils.py` inside the `client.chat.completions.create(...)` call:

```python
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    ...
)
```

For translation, the default model is set in `translator.py`:

```python
DEFAULT_MODEL = "gpt-4o-mini"
```

You can also override the translation model from the command line:

```bash
python translator.py --model your-model-name
```

## Running the Experiments

Run the default English setup:

```bash
python main.py
```

Run the German natural-language prompt setup:

```bash
python main.py --eval-csv data/german_test.csv --train-csv data/german_train.csv --natural-prompt-lang de
```

The default configuration is defined near the top of `main.py`:

```python
EVAL_CSV_PATH_DEFAULT = "data/test.csv"
TRAIN_CSV_PATH_DEFAULT = "data/train.csv"
USE_ZERO_SHOT = True
USE_CODE_PROMPT = False
```

To run few-shot experiments, set `USE_ZERO_SHOT = False`. To use code-style prompting, set `USE_CODE_PROMPT = True`.

Few-shot sample counts are controlled by:

```python
DYNAMIC_POSITIVE_SAMPLES = 1
DYNAMIC_NA_SAMPLES = 2
SEMANTIC_SIMILARITY_SAMPLES = 2
```

## Evaluation

After inference, predictions are written to:

```text
artifacts/results/results.csv
```

Run evaluation with:

```bash
python evaluation.py
```

`evaluation.py` computes:

- Binary F1, where all non-`NA` labels are treated as positive relations.
- Multi-class micro F1 over the relation labels.

Evaluation outputs are written to:

- `artifacts/metrics/evaluation_log.csv`
- `artifacts/metrics/evaluation_summary.csv`
- `artifacts/metrics/evaluation_average.csv`

When few-shot logging is enabled, selected examples are written to:

```text
artifacts/logs/few_shot_log.csv
```

## Results and Artifacts

The `artifacts/` directory contains generated outputs such as model predictions, metric summaries, few-shot logs. These files are intentionally ignored by Git because they are run outputs and can become large or change frequently.
