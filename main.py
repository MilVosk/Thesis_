#from utils.data_loader import get_dataframe
from utils.generate_prompt_examples import *
from utils.prompt_generator import prompt_generator
from utils.gpt_utils import generate_gpt_response_with_relations, parse_multiple_responses
import pandas as pd


def get_dataframe(file, drop_first_row=False):
    df = pd.read_csv(file, names=["label", "text"])
    if drop_first_row:
        df = df.drop(index=df.index[0])
        df = df.reset_index(drop=True)
    return df

def main():
    # Step 1: Load the training data
    df = get_dataframe('data/train.csv', drop_first_row=True)

    # Step 2: Create structured examples for prompt generation
    examples_df = get_examples_as_dataframe(df)

    # Step 3: Generate a prompt based on examples
    prompt = prompt_generator(examples_df)

    # Step 4: Load the test data
    test_df = get_dataframe('data/test.csv', drop_first_row=True)
    #print(test_df)
    # Step 5: Generate GPT responses for the test set
    result = generate_gpt_response_with_relations(prompt, test_df)
    
    #print(result)

    # Step 6: Parse the GPT responses into a structured format

    #print(parsed_results)parsed_results
    prediction_df = parse_multiple_responses(result)

    # Step 7: Save the parsed predictions
    prediction_df.to_csv('predicted_relations.csv', index=False)
    print("✅ Predictions saved to predicted_relations.csv")


if __name__ == "__main__":
    main()