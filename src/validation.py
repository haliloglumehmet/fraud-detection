# -*- coding: utf-8 -*-
"""
@author: mehme
"""
import pandas as pd


def analyze_missing_data(df):
    # eda notebookta da kullandim
    out = pd.DataFrame({
        "Missing_Count": df.isnull().sum(),
        "Missing_Percentage": df.isnull().mean() * 100,
        "Data_Type": df.dtypes,
    }).sort_values("Missing_Percentage", ascending=False)
    print("eksik veri top 5:")
    print(out.head(5))
    return out

def split_data_chronologically(df):
    # zamana gore bol, rastgele degil
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)
    train = df.iloc[:t1].copy()
    val = df.iloc[t1:t2].copy()
    test = df.iloc[t2:].copy()
    assert len(train) + len(val) + len(test) == n
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test


def verify_split_integrity(train, val, test):
    assert train["TransactionDT"].max() <= val["TransactionDT"].min()
    assert val["TransactionDT"].max() <= test["TransactionDT"].min()
    print("split ok, sizinti yok")
    if "isFraud" in train.columns:
        print(f"train fraud %{train['isFraud'].mean()*100:.2f}")
        print(f"val fraud   %{val['isFraud'].mean()*100:.2f}")
        print(f"test fraud  %{test['isFraud'].mean()*100:.2f}")
