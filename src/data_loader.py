import pandas as pd

def load_data(path="data/Mine_Dataset.xls"):
    df = pd.read_excel(path, sheet_name="Normalized_Data")
    X = df[["V", "H", "S"]].values
    y = df["M"].values
    return df, X, y