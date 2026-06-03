# -*- coding: utf-8 -*-
"""
@author: mehme
"""
import os
import sys

from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from bootstrap import init_pipeline
from agents import process_transaction
from features import build_model_matrix
from rag_pipeline import rag_reason

ROOT = os.path.dirname(__file__)

# rag testinde rule id -> policy dosyasi (kendim eslestirdim)
rule_policy = {
    "R01": "entity_trust.txt",
    "R02": "missing_field_risk.txt",
    "R03": "velocity_policy.txt",
    "R04": "entity_trust.txt",
    "R05": "gece_islemleri.txt",
    "R06": "score_thresholds.txt",
    "R07": "email_device_risk.txt",
    "R08": "gece_islemleri.txt",
}

deps, val_df, val_scored, y_val, fe_map, test_scored, y_test = init_pipeline(
    os.path.join(ROOT, "data"),
    os.path.join(ROOT, "config", "rules.yaml"),
    os.path.join(ROOT, "config", "fraud_policies"),
)


print("\nval metrikler")
x_val = build_model_matrix(fe_map["val"], deps["artf"])
proba = deps["model"].predict_proba(x_val)[:, 1]
print("LGBM PR-AUC :", round(average_precision_score(y_val, proba), 4))
print("raw PR-AUC  :", round(average_precision_score(y_val, val_scored["final_raw_anomaly_score"]), 4))
print("adj PR-AUC  :", round(average_precision_score(y_val, val_scored["final_anomaly_score"]), 4))
print("rule PR-AUC :", round(average_precision_score(y_val, val_scored["rule_final_score"]), 4))
print(val_scored["rule_decision"].value_counts())


print("\ntest metrikler")
x_test = build_model_matrix(fe_map["test"], deps["artf"])
proba_test = deps["model"].predict_proba(x_test)[:, 1]
print("LGBM PR-AUC :", round(average_precision_score(y_test, proba_test), 4))
print("raw PR-AUC  :", round(average_precision_score(y_test, test_scored["final_raw_anomaly_score"]), 4))
print("adj PR-AUC  :", round(average_precision_score(y_test, test_scored["final_anomaly_score"]), 4))
print("rule PR-AUC :", round(average_precision_score(y_test, test_scored["rule_final_score"]), 4))

print("\nfraud recall val")
toplam_fraud = int(val_scored["isFraud"].sum())
review = val_scored[val_scored["rule_decision"] == "REVIEW"]
block = val_scored[val_scored["rule_decision"] == "BLOCK"]
hepsi = val_scored[val_scored["rule_decision"].isin(["REVIEW", "BLOCK"])]

rev_fraud = int(review["isFraud"].sum())
blk_fraud = int(block["isFraud"].sum())
hepsi_fraud = int(hepsi["isFraud"].sum())

print("toplam fraud :", toplam_fraud)
print("REVIEW yakal :", rev_fraud, "/", toplam_fraud, "=", round(rev_fraud / toplam_fraud, 4))
print("BLOCK yakal  :", blk_fraud, "/", toplam_fraud, "=", round(blk_fraud / toplam_fraud, 4))
print("ikisi toplam :", hepsi_fraud, "/", toplam_fraud, "=", round(hepsi_fraud / toplam_fraud, 4))
if len(review) > 0:
    print("REVIEW icinde fraud %:", round(100 * rev_fraud / len(review), 2))
if len(block) > 0:
    print("BLOCK icinde fraud % :", round(100 * blk_fraud / len(block), 2))


print("\nrag test")
eslesen = val_scored[val_scored["matched_rules"].astype(str).str.len() > 0]
ornek = eslesen.sample(n=min(100, len(eslesen)), random_state=42)
hit = 0
miss = 0
for idx in ornek.index:
    row = ornek.loc[idx]
    rules = str(row["matched_rules"]).split(";")
    beklenen = []
    for r in rules:
        if r in rule_policy:
            beklenen.append(rule_policy[r])
    if len(beklenen) == 0:
        continue
    rag = rag_reason(row, use_llm=False)
    bulunan = []
    for h in rag["policy_snippets"]:
        bulunan.append(h["source"])
    ok = False
    for b in beklenen:
        if b in bulunan:
            ok = True
            break
    if ok:
        hit += 1
    else:
        miss += 1
print("ornek:", hit + miss, "hit:", hit, "miss:", miss)
if hit + miss > 0:
    print("hit-rate:", round(hit / (hit + miss), 4))

print("\nfraud ornek")
idx = val_scored[val_scored["isFraud"] == 1].index[0]
ctx = process_transaction(val_df.loc[[idx]], deps)
r = ctx["result_df"].iloc[0]
print(r["rule_decision"], "skor=", round(r["rule_final_score"], 3))
print(ctx["explanation"].get("explanation", "-"))
