import pandas as pd

def describe_data(df: pd.DataFrame):
    return df.describe(include='all')

def value_counts_column(df: pd.DataFrame, column: str):
    return df[column].value_counts()
