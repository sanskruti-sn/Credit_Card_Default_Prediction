import pandas as pd

def load_data(filepath='data/UCI_Credit_Card.csv'):
    df = pd.read_csv(filepath)
    return df
