import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_gpt_response_with_relations(prompt, text_df):
    text_samples = text_df["text"].tolist()
    extracted_labels = []
    for text in text_samples:
        messages = [
            {"role": "system", "content": "You are a helpful assistant for relation extraction."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Text:\n{text}"}
        ]
        completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        response = completion.choices[0].message.content.strip()
        extracted_labels.append(response)
    return extracted_labels

def parse_multiple_responses(responses):
    """
    Parses a list of response strings and returns a DataFrame with columns:
    'Head_Type', 'Tail_Type', 'Predicted_Relation', 'Relation_Present'
    """
    parsed_data = []
    
    for response in responses:
        try:
            lines = response.strip().splitlines()
            head_type, tail_type, relation = "unknown", "unknown", "error"

            for line in lines:
                if "Head entity type:" in line:
                    head_type = line.split("Head entity type:")[1].strip()
                elif "Tail entity type:" in line:
                    tail_type = line.split("Tail entity type:")[1].strip()
                elif "Relation type:" in line:
                    relation = line.split("Relation type:")[1].strip()

            parsed_data.append({
                "Head_Type": head_type,
                "Tail_Type": tail_type,
                "Predicted_Relation": relation,
                "Relation_Present": 0 if relation == "no relation" else 1
            })

        except Exception as e:
            parsed_data.append({
                "Head_Type": "unknown",
                "Tail_Type": "unknown",
                "Predicted_Relation": "error",
                "Relation_Present": -1
            })

    return pd.DataFrame(parsed_data)

def run_relation_extraction(text_samples, prompt):
    results = []
    for text in text_samples:
        messages = [
            {"role": "system", "content": "You are a helpful assistant for relation extraction."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Text:\n{text}"}
        ]
        completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        response = completion.choices[0].message.content.strip()
        results.append(parse_and_store_response(response, text))
    return pd.DataFrame(results)