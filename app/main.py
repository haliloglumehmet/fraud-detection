# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import os
import sys

from fastapi import FastAPI, HTTPException

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from bootstrap import init_pipeline
from agents import process_transaction
from rag_pipeline import rag_reason

app = FastAPI(title="fraud api")
deps = None
val_df = None


@app.on_event("startup")
def startup():
    global deps, val_df
    print("api basliyor, model train olacak...")
    data_path = os.path.join(ROOT, "data")
    rules_path = os.path.join(ROOT, "config", "rules.yaml")
    policy_path = os.path.join(ROOT, "config", "fraud_policies")
    deps, val_df, _, _, _, _, _ = init_pipeline(data_path, rules_path, policy_path)


def find_row(tx_id):
    row = val_df[val_df["TransactionID"] == tx_id]
    if len(row) == 0:
        raise HTTPException(status_code=404, detail="id bulunamadi")
    return row


def run_tx(tx_id):
    row = find_row(tx_id)
    ctx = process_transaction(row, deps)
    r = ctx["result_df"].iloc[0]
    return ctx, r


@app.post("/score")
def score(body: dict):
    tx_id = body["transaction_id"]
    _, r = run_tx(tx_id)
    return {
        "transaction_id": tx_id,
        "final_anomaly_score": float(r["final_anomaly_score"]),
        "rule_final_score": float(r["rule_final_score"]),
        "decision": r["rule_decision"],
    }


@app.post("/explain")
def explain(body: dict):
    tx_id = body["transaction_id"]
    ctx, r = run_tx(tx_id)
    exp = ctx["explanation"].get("explanation", "")
    return {
        "transaction_id": tx_id,
        "decision": r["rule_decision"],
        "score": float(r["rule_final_score"]),
        "rules": r.get("rule_explanations", ""),
        "explanation": exp,
        "messages": ctx["messages"],
    }


@app.post("/rules/evaluate")
def rules_evaluate(body: dict):
    tx_id = body["transaction_id"]
    _, r = run_tx(tx_id)
    return {
        "transaction_id": tx_id,
        "decision": r["rule_decision"],
        "matched_rules": r.get("matched_rules", ""),
        "rule_explanations": r.get("rule_explanations", ""),
        "rule_score_boost": float(r.get("rule_score_boost", 0)),
        "rule_final_score": float(r["rule_final_score"]),
    }


@app.post("/rag/query")
def rag_query(body: dict):
    tx_id = body["transaction_id"]
    query = body.get("query")
    _, r = run_tx(tx_id)
    use_llm = True
    if deps and "use_llm" in deps:
        use_llm = deps["use_llm"]
    return rag_reason(r, query=query, use_llm=use_llm)
