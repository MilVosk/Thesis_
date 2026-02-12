def prompt_generator(examples_df):
    prompt = [
        (
            "You are an information extraction model specialized in biodiversity texts.\n"
            "Task: Given ONE sentence containing annotated entities, decide whether the sentence explicitly states a semantic relation between the entity mentions.\n"
            "Entities appear as @ENTITY_TYPE$ where ENTITY_TYPE ∈ {ORGANISM, ENVIRONMENT, PHENOMENA, QUALITY, LOCATION, MATTER}.\n"
            "OUTPUT FORMAT (strict):\n"
            
            "- If a relation is present: 1, HAVE | 1, OCCUR_IN | 1, INFLUENCE\n"
            "- If no relation is present: 0, NA\n"
            "Return ONLY the final label line. No extra text.\n"
            "If you output '1', you must explicitly name the relation label (HAVE, OCCUR_IN, or INFLUENCE); never leave it blank.\n"
            
            "IMPORTANT DECISION RULES:\n"
            "1) First decide if the sentence contains an explicit relation statement between the named entities.\n"
            "- Co-occurrence is NOT enough.\n"
            "- Carefully read the sentence to detect, wether there is relation between the entities or not.\n"
            "2) If (and only if) an explicit link exists, choose exactly ONE relation label:\n"
            "A) OCCUR_IN\n"
            "Choose OCCUR_IN if one entity is stated to be located in / found in / present in / recorded in / distributed in a place or setting.\n"
            "Allowed type pairs (order irrelevant): MATTER-ENVIRONMENT, ENVIRONMENT-MATTER, ORGANISM-LOCATION, LOCATION-ORGANISM,\n"
            "ORGANISM-ORGANISM, PHENOMENA-LOCATION, LOCATION-PHENOMENA, ENVIRONMENT-LOCATION, LOCATION-ENVIRONMENT.\n"
            "B) HAVE\n"
            "Choose HAVE if one entity is a property, attribute, component, part, measurement, or state of another.\n"
            "Allowed type pairs (order irrelevant): ORGANISM-ENVIRONMENT, ORGANISM-QUALITY, ENVIRONMENT-QUALITY, MATTER-QUALITY,\n"
            "ENVIRONMENT-ORGANISM, QUALITY-ORGANISM, QUALITY-ENVIRONMENT, QUALITY-MATTER.\n"
            "C) INFLUENCE\n"
            "Choose INFLUENCE only if the sentence states that changing one entity has an impact on the other.\n"
            "Allowed type pairs (order irrelevant): ORGANISM-PHENOMENA, PHENOMENA-ORGANISM, ORGANISM-MATTER, MATTER-ORGANISM,\n"
            "PHENOMENA-PHENOMENA, PHENOMENA-QUALITY, QUALITY-PHENOMENA, PHENOMENA-ENVIRONMENT, ENVIRONMENT-PHENOMENA, QUALITY-QUALITY.\n"
            "4)Mere correlation, association, or background description is NOT sufficient.\n"
            "5) Direction is irrelevant.\n"
            "6)If the annotated entities are far apart in the sentence (e.g., separated by long clauses or >20 tokens) prefer 0, NA unless the text states an explicit relation.\n"
            "Return exactly one line in the required schema.\n"

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
