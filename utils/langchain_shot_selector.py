from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Set
import math

import pandas as pd

try:
    from langchain.prompts.example_selector.base import BaseExampleSelector
    from langchain_community.document_loaders import CSVLoader

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    class BaseExampleSelector:                
        """Fallback base selector when LangChain is unavailable."""

        def add_example(self, example: Dict[str, str]) -> None:
            raise NotImplementedError

        def select_examples(
            self, input_variables: Mapping[str, str] | None = None
        ) -> List[Dict[str, str]]:
            raise NotImplementedError

    CSVLoader = None                                 

try:

    from langchain.prompts.example_selector.semantic_similarity import (
        SemanticSimilarityExampleSelector,
    )
except ImportError:                                          
    try:

        from langchain_community.example_selectors.semantic_similarity import (                
            SemanticSimilarityExampleSelector,
        )
    except ImportError:                                          
        SemanticSimilarityExampleSelector = None                      

try:
    from langchain_community.vectorstores import FAISS
except ImportError:                                          
    FAISS = None                      

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:                                          
    OpenAIEmbeddings = None                      


ENTITY_TOKEN_PATTERN = re.compile(r"@([A-Z_]+)\$")
LABEL_KEYWORDS: Dict[str, List[str]] = {
    "INFLUENCE": ["influence", "affect", "drive", "impact", "alter"],
    "OCCUR_IN": ["occur", "present", "found in", "inhabit", "located"],
    "HAVE": ["has", "have", "possess", "contain"],
}


def _extract_entity_pair(text: str | None) -> Tuple[str, str] | None:
    normalized = "" if text is None else str(text)
    matches = ENTITY_TOKEN_PATTERN.findall(normalized)
    if len(matches) < 2:
        return None
    return matches[0], matches[1]


def _extract_entity_pairs(text: str | None) -> List[Tuple[str, str]]:
    normalized = "" if text is None else str(text)
    matches = ENTITY_TOKEN_PATTERN.findall(normalized)
    if len(matches) < 2:
        return []
    return [
        (matches[i], matches[j])
        for i in range(len(matches) - 1)
        for j in range(i + 1, len(matches))
    ]


def _load_examples_with_langchain(
    source_csv: str | Path,
    *,
    label_column: str,
    text_column: str,
    has_header: bool,
) -> pd.DataFrame:
    if not LANGCHAIN_AVAILABLE:
        read_kwargs = {
            "header": 0 if has_header else None,
            "keep_default_na": False,
        }
        if not has_header:
            read_kwargs["names"] = [label_column, text_column]
            read_kwargs["usecols"] = [0, 1]
        return pd.read_csv(source_csv, **read_kwargs)

    csv_args = {"delimiter": ","}
    if not has_header:
        csv_args["fieldnames"] = [label_column, text_column]

    loader = CSVLoader(
        file_path=str(source_csv),
        csv_args=csv_args,
        encoding="utf-8",
    )
    documents = loader.load()

    records: List[Dict[str, str]] = []
    for doc in documents:
        row: Dict[str, str] = {}
        for line in doc.page_content.splitlines():
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            row[key.strip()] = value.rstrip("\n")

        if label_column not in row or text_column not in row:
            continue

        records.append(
            {
                label_column: row[label_column].strip(),
                text_column: row[text_column],
            }
        )

    if not records:
        raise ValueError(
            f"No records could be loaded via LangChain from {source_csv}. "
            "Please check the CSV format."
        )

    return pd.DataFrame.from_records(records)


@dataclass
class LabelBalancedExampleSelector(BaseExampleSelector):
    examples: Sequence[Dict[str, str]]
    label_field: str
    samples_per_label: int
    random_seed: int | None = None

    def __post_init__(self) -> None:
        grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for example in self.examples:
            label = str(example[self.label_field])
            grouped[label].append(example)
        rng = random.Random(self.random_seed)
        for label_examples in grouped.values():
            rng.shuffle(label_examples)
        self._grouped_examples = grouped

    def add_example(self, example: Dict[str, str]) -> None:
        label = str(example.get(self.label_field, ""))
        self._grouped_examples.setdefault(label, []).append(example)

    def select_examples(self, _: Mapping[str, str] | None = None) -> List[Dict[str, str]]:
        selected: List[Dict[str, str]] = []
        for label in sorted(self._grouped_examples):
            label_examples = self._grouped_examples[label]
            selected.extend(label_examples[: self.samples_per_label])
        return selected


def select_few_shot_examples(
    source_csv: str | Path,
    label_column: str = "gold",
    text_column: str = "text",
    samples_per_label: int = 5,
    has_header: bool = True,
    random_seed: int | None = 13,
) -> pd.DataFrame:

    df = _load_examples_with_langchain(
        source_csv,
        label_column=label_column,
        text_column=text_column,
        has_header=has_header,
    )

    if label_column not in df.columns or text_column not in df.columns:
        raise ValueError(
            f"CSV at {source_csv} must contain '{label_column}' and '{text_column}' columns."
        )

    examples = df[[label_column, text_column]].to_dict("records")
    selector = LabelBalancedExampleSelector(
        examples=examples,
        label_field=label_column,
        samples_per_label=samples_per_label,
        random_seed=random_seed,
    )

    selected_examples = selector.select_examples()
    return pd.DataFrame(selected_examples)


@dataclass
class EntityPairExampleSelector(BaseExampleSelector):
    """Select examples whose first two entity tags match the query text."""

    samples_per_pair: int
    examples_by_pair: Dict[Tuple[str, str], List[Dict[str, str]]]
    fallback_examples: List[Dict[str, str]]
    random_seed: int | None = None

    def __post_init__(self) -> None:
        rng = random.Random(self.random_seed)
        for examples in self.examples_by_pair.values():
            rng.shuffle(examples)
        rng.shuffle(self.fallback_examples)

    def add_example(self, example: Dict[str, str]) -> None:
        text_value = str(example.get("text", ""))
        pair = _extract_entity_pair(text_value)
        if pair is not None:
            self.examples_by_pair.setdefault(pair, []).append(example)
        else:
            self.fallback_examples.append(example)

    def select_examples(self, input_variables: Mapping[str, str] | None = None) -> List[Dict[str, str]]:
        text = "" if input_variables is None else input_variables.get("text", "")
        pair = _extract_entity_pair(text)
        selected: List[Dict[str, str]] = []

        if pair and pair in self.examples_by_pair:
            selected.extend(self.examples_by_pair[pair][: self.samples_per_pair])

        if len(selected) < self.samples_per_pair:
            for candidate in self.fallback_examples:
                if candidate in selected:
                    continue
                selected.append(candidate)
                if len(selected) >= self.samples_per_pair:
                    break

        return selected

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        label_column: str = "gold",
        text_column: str = "text",
        label_filter: Callable[[str], bool] | None = None,
        samples_per_pair: int = 5,
        random_seed: int | None = None,
    ) -> "EntityPairExampleSelector":
        if label_filter is not None:
            mask = df[label_column].apply(lambda value: label_filter(str(value)))
            filtered_df = df[mask]
        else:
            filtered_df = df

        if filtered_df.empty:
            raise ValueError("No rows available to build the entity-pair selector.")

        grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
        fallback: List[Dict[str, str]] = []

        for _, row in filtered_df.iterrows():
            text_value = str(row[text_column])
            entry = {"gold": row[label_column], "text": text_value}
            pair = _extract_entity_pair(text_value)
            if pair is not None:
                grouped[pair].append(entry)
            fallback.append(entry)

        return cls(
            samples_per_pair=samples_per_pair,
            examples_by_pair=grouped,
            fallback_examples=fallback,
            random_seed=random_seed,
        )


def build_entity_pair_selector(
    source_csv: str | Path,
    *,
    label_column: str = "gold",
    text_column: str = "text",
    samples_per_pair: int = 5,
    has_header: bool = False,
    random_seed: int | None = 13,
    label_filter: Callable[[str], bool] | None = None,
) -> Optional[EntityPairExampleSelector]:
    read_kwargs = {
        "header": 0 if has_header else None,
        "keep_default_na": False,
    }
    if not has_header:
        read_kwargs["names"] = [label_column, text_column]
        read_kwargs["usecols"] = [0, 1]

    df = pd.read_csv(source_csv, **read_kwargs)

    if label_column not in df.columns or text_column not in df.columns:
        raise ValueError(
            f"CSV at {source_csv} must contain '{label_column}' and '{text_column}' columns."
        )

    try:
        return EntityPairExampleSelector.from_dataframe(
            df,
            label_column=label_column,
            text_column=text_column,
            samples_per_pair=samples_per_pair,
            random_seed=random_seed,
            label_filter=label_filter,
        )
    except ValueError as exc:
        print(
            "Warning: unable to build entity-pair selector (" + str(exc) + ")."
            " Continuing without entity-pair examples."
        )
        return None


def build_na_entity_pair_selector(
    source_csv: str | Path,
    *,
    label_column: str = "gold",
    text_column: str = "text",
    samples_per_pair: int = 5,
    na_token: str = "NA",
    has_header: bool = False,
    random_seed: int | None = 13,
) -> Optional[EntityPairExampleSelector]:
    normalized_na = na_token.strip().upper()
    return build_entity_pair_selector(
        source_csv,
        label_column=label_column,
        text_column=text_column,
        samples_per_pair=samples_per_pair,
        has_header=has_header,
        random_seed=random_seed,
        label_filter=lambda label: label.strip().upper() == normalized_na,
    )


def build_relation_entity_pair_selector(
    source_csv: str | Path,
    *,
    label_column: str = "gold",
    text_column: str = "text",
    samples_per_pair: int = 5,
    na_token: str = "NA",
    has_header: bool = False,
    random_seed: int | None = 13,
) -> Optional[EntityPairExampleSelector]:
    normalized_na = na_token.strip().upper()
    return build_entity_pair_selector(
        source_csv,
        label_column=label_column,
        text_column=text_column,
        samples_per_pair=samples_per_pair,
        has_header=has_header,
        random_seed=random_seed,
        label_filter=lambda label: label.strip().upper() != normalized_na,
    )


@dataclass
class BalancedEntityPairSelector(BaseExampleSelector):
    """Return equal numbers of NA and positive samples for matching entity pairs."""

    positive_samples: int
    na_samples: int
    examples_by_pair: Dict[
        Tuple[str, str], Dict[str, List[Dict[str, str]]]
    ]
    positive_fallback: List[Dict[str, str]]
    na_fallback: List[Dict[str, str]]
    positive_by_label: Dict[str, List[Dict[str, str]]]
    pair_label_candidates: Dict[Tuple[str, str], List[str]]
    unordered_label_candidates: Dict[frozenset[str], List[str]]
    random_seed: int | None = None
    max_pairs: int | None = 2
    max_total_examples: int | None = 12
    allow_duplicates: bool = True

    def __post_init__(self) -> None:
        rng = random.Random(self.random_seed)
        for pair_dict in self.examples_by_pair.values():
            rng.shuffle(pair_dict["positive"])
            rng.shuffle(pair_dict["na"])
        rng.shuffle(self.positive_fallback)
        rng.shuffle(self.na_fallback)
        for examples in self.positive_by_label.values():
            rng.shuffle(examples)

    def add_example(self, example: Dict[str, str]) -> None:
        text_value = str(example.get("text", ""))
        label_value = str(example.get("gold", ""))
        pair = _extract_entity_pair(text_value)
        if pair is None:
            return

        bucket = self.examples_by_pair.setdefault(
            pair, {"positive": [], "na": []}
        )
        normalized_label = label_value.strip().upper()
        entry = {"gold": normalized_label, "text": text_value}
        if normalized_label == "NA":
            bucket["na"].append(entry)
            self.na_fallback.append(entry)
        else:
            bucket["positive"].append(entry)
            self.positive_fallback.append(entry)
            self.positive_by_label.setdefault(normalized_label, []).append(entry)
            candidates = self.pair_label_candidates.setdefault(pair, [])
            if normalized_label not in candidates:
                candidates.append(normalized_label)
            unordered = frozenset(pair)
            unordered_candidates = self.unordered_label_candidates.setdefault(
                unordered, []
            )
            if normalized_label not in unordered_candidates:
                unordered_candidates.append(normalized_label)

    def _draw(
        self,
        pool: List[Dict[str, str]],
        count: int,
        fallback: List[Dict[str, str]],
        selected_keys: Set[Tuple[str, str]],
        remaining: int | None,
    ) -> List[Dict[str, str]]:
        selection: List[Dict[str, str]] = []
        if self.allow_duplicates:

            combined = pool if pool else fallback
            if not combined:
                return selection
            idx = 0
            while len(selection) < count:
                candidate = combined[idx % len(combined)]
                selection.append(candidate)
                idx += 1
                if remaining is not None and len(selection) >= remaining:
                    break
            return selection

        for candidate in pool:
            key = (candidate["gold"], candidate["text"])
            if key in selected_keys:
                continue
            selection.append(candidate)
            selected_keys.add(key)
            if len(selection) >= count:
                return selection
            if remaining is not None and len(selection) >= remaining:
                return selection
        for candidate in fallback:
            key = (candidate["gold"], candidate["text"])
            if key in selected_keys:
                continue
            selection.append(candidate)
            selected_keys.add(key)
            if len(selection) >= count:
                break
            if remaining is not None and len(selection) >= remaining:
                break
        return selection

    def select_examples(
        self, input_variables: Mapping[str, str] | None = None
    ) -> List[Dict[str, str]]:
        text = "" if input_variables is None else input_variables.get("text", "")
        pairs = _extract_entity_pairs(text)
        if not pairs:
            pairs = [None]

        unique_pairs: List[Tuple[str, str] | None] = []
        seen: Set[Tuple[str, str] | None] = set()
        for pair in pairs:
            if pair not in seen:
                unique_pairs.append(pair)
                seen.add(pair)
            if self.max_pairs is not None and len(unique_pairs) >= self.max_pairs:
                break

        selected: List[Dict[str, str]] = []
        selected_keys: Set[Tuple[str, str]] = set()

        for pair in unique_pairs:
            if (
                self.max_total_examples is not None
                and len(selected) >= self.max_total_examples
            ):
                break

            bucket = self.examples_by_pair.get(pair) if pair else None
            positives = bucket["positive"] if bucket else []
            negatives = bucket["na"] if bucket else []
            inferred_labels = self._infer_labels_for_pair(pair, text)
            if not inferred_labels:
                inferred_labels = [None]

            remaining = (
                None
                if self.max_total_examples is None
                else self.max_total_examples - len(selected)
            )
            positive_budget = self.positive_samples
            for idx, label in enumerate(inferred_labels):
                if positive_budget <= 0:
                    break
                slots_left = len(inferred_labels) - idx
                per_label_quota = max(1, math.ceil(positive_budget / slots_left))
                filtered_positive_pool = (
                    [ex for ex in positives if ex["gold"] == label]
                    if label and positives
                    else positives
                )
                fallback_pool = (
                    self.positive_by_label.get(label, self.positive_fallback)
                    if label
                    else self.positive_fallback
                )
                drawn = self._draw(
                    filtered_positive_pool,
                    per_label_quota,
                    fallback_pool,
                    selected_keys,
                    remaining,
                )
                selected.extend(drawn)
                positive_budget -= len(drawn)
                remaining = (
                    None
                    if self.max_total_examples is None
                    else self.max_total_examples - len(selected)
                )
                if remaining is not None and remaining <= 0:
                    break

            remaining = (
                None
                if self.max_total_examples is None
                else self.max_total_examples - len(selected)
            )
            if remaining is not None and remaining <= 0:
                break
            selected.extend(
                self._draw(
                    negatives,
                    self.na_samples,
                    self.na_fallback,
                    selected_keys,
                    remaining,
                )
            )

        return selected

    def _infer_labels_for_pair(
        self, pair: Tuple[str, str] | None, text: str
    ) -> List[str]:
        if pair is None:
            return []
        candidates = self.pair_label_candidates.get(pair)
        if not candidates:
            unordered = frozenset(pair)
            candidates = self.unordered_label_candidates.get(unordered, [])
        if not candidates:
            return []
        return self._apply_context_bias(text, candidates)

    def _apply_context_bias(self, text: str, candidates: List[str]) -> List[str]:
        lower_text = text.lower()
        scored: List[Tuple[str, int]] = []
        for label in candidates:
            keywords = LABEL_KEYWORDS.get(label, [])
            score = 0
            for keyword in keywords:
                score += lower_text.count(keyword)
            scored.append((label, score))
        if any(score > 0 for _, score in scored):
            scored.sort(key=lambda item: (-item[1], candidates.index(item[0])))
            return [label for label, _ in scored]
        return candidates


def build_balanced_entity_pair_selector(
    source_csv: str | Path,
    *,
    label_column: str = "gold",
    text_column: str = "text",
    positive_samples: int = 2,
    na_samples: int = 2,
    na_token: str = "NA",
    has_header: bool = False,
    random_seed: int | None = 13,
    allow_duplicates: bool = True,
) -> Optional[BalancedEntityPairSelector]:
    df = _load_examples_with_langchain(
        source_csv,
        label_column=label_column,
        text_column=text_column,
        has_header=has_header,
    )
    normalized_na = na_token.strip().upper()
    grouped: Dict[Tuple[str, str], Dict[str, List[Dict[str, str]]]] = defaultdict(
        lambda: {"positive": [], "na": []}
    )
    fallback_positive: List[Dict[str, str]] = []
    fallback_na: List[Dict[str, str]] = []
    positive_by_label: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    pair_label_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    unordered_label_counts: Dict[frozenset[str], Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for _, row in df.iterrows():
        text_value = str(row[text_column])
        pairs = _extract_entity_pairs(text_value)
        if not pairs:
            continue

        label_value = str(row[label_column]).strip()
        normalized_label = label_value.upper()
        entry = {"gold": normalized_label, "text": text_value}

        for pair in pairs:
            if normalized_label == normalized_na:
                grouped[pair]["na"].append(entry)
            else:
                grouped[pair]["positive"].append(entry)
                pair_label_counts[pair][normalized_label] += 1
                unordered_label_counts[frozenset(pair)][normalized_label] += 1

        if normalized_label == normalized_na:
            fallback_na.append(entry)
        else:
            fallback_positive.append(entry)
            positive_by_label[normalized_label].append(entry)

    if not grouped:
        raise ValueError("Could not build entity-pair groups from training data.")
    if not fallback_positive or not fallback_na:
        raise ValueError("Need at least one NA and one positive example to build selectors.")

    def _sorted_labels(counts: Dict[str, int]) -> List[str]:
        return [
            label
            for label, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    pair_label_candidates = {
        pair: _sorted_labels(counts) for pair, counts in pair_label_counts.items()
    }
    unordered_label_candidates = {
        pair: _sorted_labels(counts)
        for pair, counts in unordered_label_counts.items()
    }

    return BalancedEntityPairSelector(
        positive_samples=positive_samples,
        na_samples=na_samples,
        examples_by_pair=grouped,
        positive_fallback=fallback_positive,
        na_fallback=fallback_na,
        positive_by_label=positive_by_label,
        pair_label_candidates=pair_label_candidates,
        unordered_label_candidates=unordered_label_candidates,
        random_seed=random_seed,
        allow_duplicates=allow_duplicates,
    )


def build_semantic_similarity_selector(
    source_csv: str | Path,
    *,
    label_column: str = "gold",
    text_column: str = "text",
    has_header: bool = False,
    top_k: int = 4,
    embeddings: Any | None = None,
    vectorstore_cls: Any | None = None,
) -> Optional[BaseExampleSelector]:
    """Construct a semantic-similarity selector backed by LangChain."""

    if SemanticSimilarityExampleSelector is None:
        raise ImportError(
            "SemanticSimilarityExampleSelector requires langchain with "
            "semantic similarity utilities installed."
        )

    if embeddings is None:
        if OpenAIEmbeddings is None:
            raise ImportError(
                "OpenAIEmbeddings unavailable. Provide custom embeddings via "
                "the 'embeddings' argument."
            )
        embeddings = OpenAIEmbeddings()

    if vectorstore_cls is None:
        if FAISS is None:
            raise ImportError(
                "FAISS vector store unavailable. Supply 'vectorstore_cls' with a "
                "LangChain-compatible implementation."
            )
        vectorstore_cls = FAISS

    df = _load_examples_with_langchain(
        source_csv,
        label_column=label_column,
        text_column=text_column,
        has_header=has_header,
    )
    if df.empty:
        raise ValueError("No rows available to build semantic similarity selector.")

    examples = df[[label_column, text_column]].to_dict("records")
    selector = SemanticSimilarityExampleSelector.from_examples(
        examples=examples,
        embeddings=embeddings,
        vectorstore_cls=vectorstore_cls,
        k=top_k,
    )
    return selector
