import pandas as pd
import re

# Define relation types and expected head/tail entity types
RELATIONS = {
    "OCCUR_IN": ("ORGANISM", "ENVIRONMENT"),
    "INFLUENCE": ("ORGANISM", "PHENOMENA"),
    "HAVE/OF": ("QUALITY", "ENVIRONMENT"),
}

def get_dataframe(file_name):
    """
    Loads the CSV file, assigns column names, removes the first row (assumed header), and returns the DataFrame.
    """
    df = pd.read_csv(file_name, names=["label", "text"])
    df = df.drop(index=df.index[0])  # Drop potential header row
    df = df.dropna(subset=["label", "text"])  # Ensure no missing values
    return df

def extract_entities(text):
    """
    Extracts entity types from the text using markers like @ENTITY$.
    Returns a list of entity types found.
    """
    return re.findall(r'@([A-Z_]+)\$', text)

def get_examples_as_dataframe(df):
    """
    Extracts 5 positive and 5 negative examples for each defined relation type.
    Returns a DataFrame with: ['text', 'relation', 'head_type', 'tail_type', 'label']
    """
    POS_LIMIT = 10
    NEG_LIMIT = 10

    positive = {rel: [] for rel in RELATIONS}
    negative = {rel: [] for rel in RELATIONS}

    for _, row in df.iterrows():
        raw_text = str(row['text']).strip()
        try:
            label = int(row['label'])
        except (ValueError, TypeError):
            continue

        entity_types = extract_entities(raw_text)
        if len(entity_types) < 2:
            continue

        head_type, tail_type = entity_types[0], entity_types[1]

        for rel, (expected_head, expected_tail) in RELATIONS.items():
            if head_type == expected_head and tail_type == expected_tail:
                example = {
                    "text": raw_text,
                    "relation": rel,
                    "head_type": head_type,
                    "tail_type": tail_type,
                    "label": "positive" if label == 1 else "negative"
                }

                if label == 1 and len(positive[rel]) < POS_LIMIT:
                    positive[rel].append(example)
                elif label == 0 and len(negative[rel]) < NEG_LIMIT:
                    negative[rel].append(example)

        # Stop early if all examples are collected
        if all(len(positive[rel]) >= POS_LIMIT and len(negative[rel]) >= NEG_LIMIT for rel in RELATIONS):
            break

    # Combine and flatten all examples
    all_examples = [ex for examples in (positive.values(), negative.values()) for ex_list in examples for ex in ex_list]
    return pd.DataFrame(all_examples)
