import pandas as pd

def get_dataframe(file, drop_first_row=False):
    
    # ='data/train.csv'
    """
    Loads the CSV file, optionally drops the first row, and assigns column names.
    
    Args:
    - file (str): Path to the CSV file.
    - drop_first_row (bool): If True, the first row will be dropped (useful if the file contains an unwanted header).
    
    Returns:
    - DataFrame: A pandas DataFrame with the desired structure.
    """
    df = pd.read_csv(file, names=["label", "text"])
    
    if drop_first_row:
        # Drop the first row and reset the index
        df = df.drop(index=df.index[0])
        df.reset_index(drop=True, inplace=True)
        
    return df