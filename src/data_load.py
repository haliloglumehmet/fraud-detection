# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import os
import numpy as np
import pandas as pd


def shrink_memory(df, name="df"):
    # ram icin dtype kucultme (590k satir icin lazim)
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            lo = df[col].min()
            hi = df[col].max()
            if pd.isna(lo) or pd.isna(hi):
                continue
            tip = str(df[col].dtype)
            if tip[:3] == "int":
                if lo > -128 and hi < 127:
                    df[col] = df[col].astype(np.int8)
                elif lo > -32768 and hi < 32767:
                    df[col] = df[col].astype(np.int16)
                elif lo > -2147483648 and hi < 2147483647:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype("category")

    end = df.memory_usage(deep=True).sum() / 1024**2
    print("[" + name + "] bellek " + str(int(start)) + " -> " + str(int(end)) + " MB")
    return df


def load_and_merge_data(data_dir):
    tx_path = os.path.join(data_dir, "train_transaction.csv")
    id_path = os.path.join(data_dir, "train_identity.csv")

    if not os.path.exists(tx_path):
        raise FileNotFoundError("train_transaction yok: " + tx_path)
    if not os.path.exists(id_path):
        raise FileNotFoundError("train_identity yok: " + id_path)

    print("csv okunuyor...")
    tx = pd.read_csv(tx_path)
    idf = pd.read_csv(id_path)

    tx = shrink_memory(tx, "train_transaction")
    idf = shrink_memory(idf, "train_identity")

    # identityde ayni TransactionID iki kez olmamali
    if idf["TransactionID"].duplicated().any():
        raise ValueError("identityde duplicate TransactionID var")

    n = len(tx)
    merged = pd.merge(tx, idf, on="TransactionID", how="left")
    if len(merged) != n:
        raise ValueError("merge sonrasi satir sayisi degisti")

    return merged
