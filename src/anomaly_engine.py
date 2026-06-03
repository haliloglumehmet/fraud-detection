# -*- coding: utf-8 -*-
"""
@author: mehme
"""
import numpy as np
import pandas as pd


def column_score(df, artf=None):
    # IQR ile tutar anomalisi
    amt = df["TransactionAmt"].copy()
    amt[amt < 0.01] = 0.01

    if artf is not None and "amt_upper_bound" in artf:
        ub = artf["amt_upper_bound"]
    else:
        q1 = amt.quantile(0.25)
        q3 = amt.quantile(0.75)
        iqr = q3 - q1
        ub = q3 + 3.0 * iqr
        if artf is not None:
            artf["amt_upper_bound"] = ub

    sc = pd.Series(0.0, index=df.index)
    yuksek = amt > ub
    sc.loc[yuksek] = 1.0 - (ub / amt.loc[yuksek])
    return sc.fillna(0.0)


def multivariate_score(df, model, cols):
    use = []
    for c in cols:
        if c in df.columns:
            use.append(c)
    if len(use) == 0:
        return pd.Series(0.0, index=df.index)

    x = df[use]
    num_iter = getattr(model, "best_iteration", None)
    if num_iter:
        p = model.predict_proba(x, num_iteration=num_iter)[:, 1]
    else:
        p = model.predict_proba(x)[:, 1]

    p[p < 0] = 0
    p[p > 1] = 1
    return pd.Series(p, index=df.index)


def entity_score(df):
    if "user_avg_amount" not in df.columns:
        return pd.Series(0.0, index=df.index)

    amt = df["TransactionAmt"].copy()
    avg = df["user_avg_amount"].copy()
    amt[amt < 0.01] = 0.01
    avg[avg < 0.01] = 0.01
    ratio = amt / avg - 1.0
    ratio[ratio < 0] = 0
    sc = np.tanh(ratio / 10.0)
    return pd.Series(sc, index=df.index).fillna(0.0)


def temporal_score(df):
    # edada gece 2 civari peak vardi
    if "Transaction_Hour" not in df.columns:
        return pd.Series(0.0, index=df.index)

    h = df["Transaction_Hour"]
    d1 = abs(h - 2)
    d2 = 24 - abs(h - 2)
    d = d1.copy()
    d[d2 < d1] = d2[d2 < d1]
    sc = np.exp(-(d ** 2) / 18.0)
    return pd.Series(sc, index=df.index).fillna(0.0)


def run_anomaly_pipeline(df, model, cols, artf=None):
    out = df.copy()
    out["score_column"] = column_score(out, artf)
    out["score_multivariate"] = multivariate_score(out, model, cols)
    out["score_entity"] = entity_score(out)
    out["score_temporal"] = temporal_score(out)
    return out
