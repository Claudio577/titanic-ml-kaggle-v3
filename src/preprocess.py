import pandas as pd
from utils import DATA_RAW

def load_data():
    train_path = DATA_RAW / "train.csv"
    test_path = DATA_RAW / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {train_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df
