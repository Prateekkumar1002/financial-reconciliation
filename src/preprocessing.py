import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(bank_path, check_path):
    bank = pd.read_csv(bank_path)
    check = pd.read_csv(check_path)

    bank["clean_desc"] = bank["description"].apply(clean_text)
    check["clean_desc"] = check["description"].apply(clean_text)

    bank["date"] = pd.to_datetime(bank["date"])
    check["date"] = pd.to_datetime(check["date"])

    bank["amount"] = bank["amount"].astype(float).round(2)
    check["amount"] = check["amount"].astype(float).round(2)

    type_map = {"debit": "dr", "credit": "cr", "dr": "dr", "cr": "cr"}

    bank["norm_type"] = bank["type"].str.lower().map(type_map)
    check["norm_type"] = check["type"].str.lower().map(type_map)

    return bank, check