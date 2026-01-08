def prompt_generator(examples_df):
    prompt = [
        (
            "You are an information extraction model specialized in the biodiversity domain.\n"
            "Your task is to detect and extract semantic relations between annotated entities in a given text.\n"
            "Entities are annotated using the format @ENTITY_TYPE$, where ENTITY_TYPE is one of {ORGANISM, ENVIRONMENT, PHENOMENA, QUALITY, LOCATION, MATTER}.\n"
            "Always analyze the sentence itself before relying on examples. Co-occurrence of entities is not sufficient; if the sentence is metadata, a study summary, or lacks an explicit link, output '0, NA'.\n"
            "Use a two-step reasoning process:\n"
            "  1) Decide which intuitive relation best fits the sentence:\n"
            "       - INFLUENCE: one entity affects or changes the other (e.g., an ORGANISM influences a PHENOMENA).\n"
            "       - OCCUR: one entity occurs in or is present within another (e.g., a PROCESS occurs in an ENVIRONMENT).\n"
            "       - HAVE/OF: one entity is a property, component, or possession of the other (e.g., QUALITY of ORGANISM).\n"
            "     If none apply (metadata, hypotheses, or no explicit interaction), treat it as NA.\n"
            "  2) Map the chosen type to the final label deterministically: INFLUENCE → INFLUENCE, OCCUR → OCCUR_IN, HAVE/OF → HAVE, None → NA.\n"
            "Valid relation labels are HAVE, OCCUR_IN, and INFLUENCE.\n"
            "Never leave the answer blank. If a relation is present, respond exactly as '1, RELATION_NAME' using one of the valid labels in uppercase. If no relation is present, respond exactly as '0, NA'.\n"
            "Study the few-shot examples and mimic the reasoning.\n"
        
        )
    ]

    for idx, row in examples_df.iterrows():
        raw_label = str(row["gold"]).strip()
        if raw_label.upper() == "NA":
            label_text = "0, NA"
        else:
            label_text = f"1, {raw_label.upper()}"
        prompt.append(
            f'Example {idx + 1}:\n'
            f'Text: "{row["text"]}"\n'
            f"Gold label: {label_text}\n"
        )

    prompt.append("Classify the next sentence following the same output format.")
    return "\n".join(prompt)
