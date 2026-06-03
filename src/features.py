# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import numpy as np
import pandas as pd

ENTITY = ["card1", "card2", "addr1"]
EMAIL_MIN = 100
MAX_NULL = 0.35

SKIP = {"TransactionID", "isFraud", "TransactionDT", "D7", "id_31"}
FE_COLS = [
    "TransactionAmt_Log", "Transaction_Hour", "day_of_week", "is_weekend",
    "D7_is_null", "id_31_is_null", "Email_Fraud_Risk_Score", "is_mobile",
    "DeviceType_is_null", "user_avg_amount", "user_transaction_count",
    "amount_vs_user_avg", "hours_since_last_tx",
]


def pick_model_cols(df):
    num = []
    cat = []
    for col in df.columns:
        if col in SKIP:
            continue
        if col in FE_COLS:
            continue
        null_oran = df[col].isnull().mean()
        if null_oran > MAX_NULL:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num.append(col)
        else:
            cat.append(col)

    feats = ["TransactionAmt"]
    for c in FE_COLS:
        feats.append(c)
    for c in num:
        if c not in feats:
            feats.append(c)
    for c in cat:
        if c not in feats:
            feats.append(c)
    return num, cat, feats


def extract_feature_artifacts(xtrain, y):
    artf = {}
    x = xtrain.copy()
    if hasattr(y, "values"):
        x["isFraud"] = y.values
    else:
        x["isFraud"] = y

    gmean = y.mean()
    artf["global_amount_mean"] = x["TransactionAmt"].mean()

    if "R_emaildomain" in x.columns:
        es = x.groupby("R_emaildomain", dropna=False)["isFraud"].agg(["count", "mean"])
        es = es[es["count"] >= EMAIL_MIN]
        # smooth mean - az ornekli domainleri duzeltmek icin
        smooth = (es["count"] * es["mean"] + 10 * gmean) / (es["count"] + 10)
        artf["email_risk_map"] = smooth.to_dict()
        artf["global_email_mean"] = gmean

    artf["entity_amount_map"] = x.groupby(ENTITY, dropna=False)["TransactionAmt"].mean().to_dict()
    artf["entity_count_map"] = x.groupby(ENTITY, dropna=False)["TransactionID"].count().to_dict()

    num, cat, feats = pick_model_cols(x)
    artf["model_numeric_cols"] = num
    artf["model_categorical_cols"] = cat
    artf["model_feature_cols"] = feats
    return artf


def merge_entity(df, amap, col_name, fallback):
    if len(amap) == 0:
        out = pd.Series(fallback, index=df.index, dtype=np.float32)
        return out

    rows = []
    for key, val in amap.items():
        rows.append({"card1": key[0], "card2": key[1], "addr1": key[2], col_name: val})
    sf = pd.DataFrame(rows)

    m = df[ENTITY].merge(sf, on=ENTITY, how="left")
    out = m[col_name].fillna(fallback).astype(np.float32)
    out.index = df.index
    return out


def transform_features(x, artf):
    df = x.copy()
    if artf is None:
        artf = {}

    g_amt = artf.get("global_amount_mean", 0.0)
    g_email = artf.get("global_email_mean", 0.035)

    if "D7" in df.columns:
        df["D7_is_null"] = df["D7"].isnull().astype(np.int8)
    else:
        df["D7_is_null"] = 1

    if "id_31" in df.columns:
        df["id_31_is_null"] = df["id_31"].isnull().astype(np.int8)
    else:
        df["id_31_is_null"] = 1

    if "TransactionDT" in df.columns:
        df["Transaction_Hour"] = ((df["TransactionDT"] / 3600) % 24).astype(np.int8)
        df["day_of_week"] = ((df["TransactionDT"] / 86400) % 7).astype(np.int8)
        df["is_weekend"] = 0
        df.loc[df["day_of_week"].isin([5, 6]), "is_weekend"] = 1

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_Log"] = np.log1p(df["TransactionAmt"]).astype(np.float32)

    if "DeviceType" in df.columns:
        df["DeviceType_is_null"] = df["DeviceType"].isnull().astype(np.int8)
        df["is_mobile"] = 0
        df.loc[df["DeviceType"] == "mobile", "is_mobile"] = 1
    else:
        df["DeviceType_is_null"] = 1
        df["is_mobile"] = 0

    has_entity = True
    for c in ENTITY:
        if c not in df.columns:
            has_entity = False
            break

    if has_entity:
        df["user_avg_amount"] = merge_entity(
            df, artf.get("entity_amount_map", {}), "user_avg_amount", g_amt
        )
        hist = merge_entity(df, artf.get("entity_count_map", {}), "hist_count", 0.0)
        hist = hist.astype(np.int32)
        sess = df.groupby(ENTITY, dropna=False).cumcount().astype(np.int32)
        df["user_transaction_count"] = hist.fillna(0) + sess.fillna(0) + 1

        if "TransactionAmt" in df.columns:
            safe = df["user_avg_amount"].copy()
            safe[safe < 0.01] = 0.01
            df["amount_vs_user_avg"] = (df["TransactionAmt"] / safe).astype(np.float32)

        if "TransactionDT" in df.columns:
            idx = df.index
            s = df.sort_values("TransactionDT")
            gap = s.groupby(ENTITY, dropna=False)["TransactionDT"].diff() / 3600.0
            df["hours_since_last_tx"] = gap.reindex(idx).fillna(-1).astype(np.float32)

    if "R_emaildomain" in df.columns:
        risk_map = artf.get("email_risk_map", {})
        df["Email_Fraud_Risk_Score"] = df["R_emaildomain"].map(risk_map)
        df["Email_Fraud_Risk_Score"] = df["Email_Fraud_Risk_Score"].fillna(g_email)
        df["Email_Fraud_Risk_Score"] = df["Email_Fraud_Risk_Score"].astype(np.float32)

    return df


def build_model_matrix(df, artf=None, feature_cols=None):
    if feature_cols is None:
        if artf is not None:
            feature_cols = artf.get("model_feature_cols")
    if feature_cols is None:
        feature_cols = ["TransactionAmt"] + FE_COLS

    cols = []
    for c in feature_cols:
        if c in df.columns:
            cols.append(c)
    return df[cols].copy()
