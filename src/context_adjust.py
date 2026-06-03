# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import numpy as np
import pandas as pd


def extract_context_artifacts(train_fe):
    artf = {}
    artf["amt_median"] = float(train_fe["TransactionAmt"].median())
    artf["trusted_tx_count"] = float(train_fe["user_transaction_count"].quantile(0.75))
    return artf


def apply_context_adjustments(df, score_col="final_raw_anomaly_score", artf=None):
    if artf is None:
        artf = {}

    out = df.copy()
    sc = out[score_col].astype(float)
    f = pd.Series(1.0, index=out.index)

    med = artf.get("amt_median")
    trusted_n = artf.get("trusted_tx_count", 20)

    # mesai saati + dusuk tutar -> skoru biraz dusur
    if "Transaction_Hour" in out.columns and med and "TransactionAmt" in out.columns:
        mesai = (out["Transaction_Hour"] >= 9) & (out["Transaction_Hour"] <= 18)
        dusuk_tutar = out["TransactionAmt"] <= med
        if "amount_vs_user_avg" in out.columns:
            dusuk_dev = out["amount_vs_user_avg"] <= 1.5
            mask1 = mesai & dusuk_tutar & dusuk_dev
        else:
            mask1 = mesai & dusuk_tutar
        f.loc[mask1] = f.loc[mask1] * 0.85

    # hafta sonu
    if "is_weekend" in out.columns:
        hafta_sonu = out["is_weekend"] == 1
        if "amount_vs_user_avg" in out.columns:
            dusuk_dev = out["amount_vs_user_avg"] <= 1.5
            mask2 = hafta_sonu & dusuk_dev
        else:
            mask2 = hafta_sonu
        f.loc[mask2] = f.loc[mask2] * 0.90

    # guvenilir kullanici
    if "user_transaction_count" in out.columns:
        cok_islem = out["user_transaction_count"] >= trusted_n
        d7_ok = True
        id_ok = True
        dev_ok = True
        if "D7_is_null" in out.columns:
            d7_ok = out["D7_is_null"] == 0
        if "id_31_is_null" in out.columns:
            id_ok = out["id_31_is_null"] == 0
        if "amount_vs_user_avg" in out.columns:
            dev_ok = out["amount_vs_user_avg"] <= 1.3
        mask3 = cok_islem & d7_ok & id_ok & dev_ok
        f.loc[mask3] = f.loc[mask3] * 0.80

    out["context_adjust_factor"] = f
    final = sc * f
    final = final.clip(0, 1)
    out["final_anomaly_score"] = final
    return out
