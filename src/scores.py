# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import numpy as np
import pandas as pd

LAYERS = ["score_column", "score_multivariate", "score_entity", "score_temporal"]
WEIGHTS = {
    "score_column": 0.15,
    "score_multivariate": 0.45,
    "score_entity": 0.25,
    "score_temporal": 0.15,
}


def normalize_layers(df, artf=None):
    if artf is None:
        artf = {}
    if "score_bounds" not in artf:
        artf["score_bounds"] = {}

    out = df.copy()
    for col in LAYERS:
        if col not in out.columns:
            continue
        if col not in artf["score_bounds"]:
            artf["score_bounds"][col] = {
                "min": float(out[col].min()),
                "max": float(out[col].max()),
            }
        lo = artf["score_bounds"][col]["min"]
        hi = artf["score_bounds"][col]["max"]
        out[col] = (out[col] - lo) / (hi - lo + 0.00001)
        out[col] = out[col].clip(0, 1)
    return out


def risk_boost(df, base):
    m = pd.Series(1.0, index=df.index)
    if "D7_is_null" in df.columns:
        m = m + df["D7_is_null"] * 0.122
    if "id_31_is_null" in df.columns:
        m = m + df["id_31_is_null"] * 0.059
    if "Email_Fraud_Risk_Score" in df.columns:
        email_bonus = pd.Series(0.0, index=df.index)
        email_bonus[df["Email_Fraud_Risk_Score"] > 0.035] = 0.061
        m = m + email_bonus
    final = base * m
    final[final > 1.0] = 1.0
    return final


def aggregate_anomaly_scores(df, weights=None, train_artifacts=None):
    if weights is None:
        weights = WEIGHTS

    total = 0
    for v in weights.values():
        total = total + v
    w = {}
    for k, v in weights.items():
        w[k] = v / total

    out = normalize_layers(df, train_artifacts)
    base = pd.Series(0.0, index=out.index)
    for col in w:
        if col in out.columns:
            base = base + out[col] * w[col]

    out["final_raw_anomaly_score"] = risk_boost(out, base)
    return out
