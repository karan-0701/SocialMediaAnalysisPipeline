import pandas as pd

def load_youtube_titles(csv_path: str, column="title") -> list:
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    return df[column].dropna().tolist(), df
