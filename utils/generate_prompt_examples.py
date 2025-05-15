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
    df = pd.read_csv(file_name, names=["Label", "Text"])
    df = df.drop(index=df.index[0])  # Drop potential header row
    df = df.dropna(subset=["Label", "Text"])  # Ensure no missing values
    return df

def extract_entities(text):
    """
    Extracts entity types from the text using markers like @ENTITY$.
    Returns a list of entity types found.
    """
    return re.findall(r'@([A-Z_]+)\$', text)

def get_examples_as_dataframe(df):
    """
    Extracts one positive and one negative example for each defined relation type.
    Returns a DataFrame with: ['text', 'relation', 'head_type', 'tail_type', 'label']
    """
    positive = {rel: None for rel in RELATIONS}
    negative = {rel: None for rel in RELATIONS}

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

                if label == 1 and not positive[rel]:
                    positive[rel] = example
                elif label == 0 and not negative[rel]:
                    negative[rel] = example

        # Break early if all examples found
        if all(positive.values()) and all(negative.values()):
            break

    # Combine and drop Nones if any relation was missing
    all_examples = [ex for ex in (*positive.values(), *negative.values()) if ex]
    return pd.DataFrame(all_examples)

"""def main():
    df = get_dataframe('C:\\Users\\marce\\Desktop\\thesis\\relation_extraction\\data\\train.csv')
    examples_df = get_examples_as_dataframe(df)

    # Save the output
    examples_df.to_csv("relation_examples2x.csv", index=False)
    print("Saved example prompts to relation_examples2x.csv")"""

if __name__ == "__main__":
    main()
