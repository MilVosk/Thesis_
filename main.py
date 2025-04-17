import openai
import os
import pandas as pd
from openai import OpenAI
from file_reader import train
from dotenv import load_dotenv
import os

load_dotenv() 
api_key = os.getenv("OPENAI_API_KEY")

#df = pd.read_csv('train.csv', names=["Label", "Text"])
#train = (df.drop(index=df.index[[0]]))
#train.head()

def balanced_sample(data, column, n_per_class, random_state=42):
    """
    Selects a balanced sample from a DataFrame with an equal number of rows for each class
    and removes those rows from df1.

    Parameters:
        df1 (pd.DataFrame): The main DataFrame to sample from.
        column (str): The column containing binary values (e.g., 0 and 1).
        n_per_class (int): Number of rows to sample for each class.
        random_state (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: A new DataFrame with a balanced sample.
        pd.DataFrame: Updated df1 with sampled rows removed.
    """
# Step 1: Select a balanced sample
    sampled_df = (data.groupby(column, group_keys=False)
                    .apply(lambda x: x.sample(n=n_per_class, random_state=random_state))
                    .reset_index(drop=True))
    remaining_data = data.merge(sampled_df, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)

    return sampled_df, remaining_data

#sampled, test = balanced_sample(train, train['Label'], 40)


# Generate prompt using langchain
def prompt_generator(examples):
    prompt_template = "You are now extracting relations from texts in the field of biodiversity. Here are a few examples that might help you understand what the relations look like:\n\n"
    
    for _, row in examples.iterrows():
        prompt_template += f'Text: "{row["Text"]}" → Label: {row["Label"]}\n'
    
    prompt_template += "\nBased on the examples, determine whether a relation is present or not. If there is a relation between the entities, give 1 as an answer, else 0."
    
    return prompt_template

#prompt = prompt_generator(sampled)



def generate_gpt_response_with_relations(prompt, text_df, max_token_length=1000):
    """
    Sends each text sample to GPT-4o for relation extraction and returns a DataFrame 
    with predicted labels (0 or 1).

    Parameters:
        prompt (str): The initial instruction for GPT-4o.
        text_df (pd.DataFrame): DataFrame containing a 'Text' column.

    Returns:
        pd.DataFrame: A new DataFrame with texts and their predicted labels.
    """
    client = OpenAI(api_key = api_key)  # Initialize OpenAI client

    # Extract text samples
    text_samples = text_df["Text"].tolist()
    extracted_labels = []

    # Process each text sample separately
    for text in text_samples:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Help me with conducting relation extraction!"},
            {"role": "user", "content": prompt},  
            {"role": "user", "content": f"Look at the texts and return 1 if relation exists and 0 if it doesn't. Do not write any text, give either 1 or 0:\n{text}"}
        ]

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        # Extract and clean response
        label = completion.choices[0].message.content.strip()
        extracted_labels.append(label)

    # Create DataFrame with extracted labels
    result_df = pd.DataFrame({"Text": text_samples, "Predicted_Label": extracted_labels})

    return result_df

"""
result_df = generate_gpt_response_with_relations(prompt, test[41:60]['Text'])
 
result_df['Gold'] = test[41:60]['Label'].reset_index(drop=True)
result_df


true = [str(i) for i in result_df['Gold']]
predicted = [str(j) for j in result_df['Predicted_Label']]

# Calculate the F1 score
f1 = f1_score(true, predicted, average='weighted')  
print(f"F1 Score: {f1}")

result_df.to_excel("output_with_f1_0.618.xlsx", index=False)
"""