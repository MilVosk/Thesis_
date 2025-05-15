def prompt_generator(examples_df):
    prompt = (
        "You are now extracting relations from texts in the field of biodiversity.\n"
        "Entities are marked using the format @ENTITY_TYPE$.\n"
        "Determine whether a relation exists between the entities.\n\n"
    )
    for _, row in examples_df.iterrows():
        label_text = row["relation"] if row["label"] == "positive" else "no relation"
        prompt += (
            f'Text: "{row["text"]}"\n'
            f"Head entity type: {row['head_type']}\n"
            f"Tail entity type: {row['tail_type']}\n"
            f"Relation type: {label_text}\n\n"
        )
    prompt += (
        "Now classify the new text:\n"
        'Text: "<sentence>"\nHead entity type: <HEAD_TYPE>\nTail entity type: <TAIL_TYPE>\nRelation type: <RELATION_TYPE>\n'
    )
    return prompt