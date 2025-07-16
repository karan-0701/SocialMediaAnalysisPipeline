import pandas as pd
from .config import LABEL_TO_ID

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    
    columns_to_drop = ['id', 'video id', 'author', 'author id', 'annotator notes', 'Unnamed: 0.1', 'Unnamed: 0']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df = df.dropna(subset=['label'])

    df['label_num'] = df['label'].map(lambda x: LABEL_TO_ID[x.strip()])
    return df

def split_data(df):
    size = len(df)
    train_size = size // 2
    val_size = (size - train_size) // 2

    train_texts = df['text'][:train_size].tolist()
    val_texts = df['text'][train_size:train_size + val_size].tolist()
    test_texts = df['text'][train_size + val_size:].tolist()

    train_labels = df['label_num'][:train_size].tolist()
    val_labels = df['label_num'][train_size:train_size + val_size].tolist()
    test_labels = df['label_num'][train_size + val_size:].tolist()

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
