# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import pandas as pd

from features import transform_features
from anomaly_engine import run_anomaly_pipeline
from scores import aggregate_anomaly_scores
from context_adjust import apply_context_adjustments
from rule_engine import evaluate_rules
from rag_pipeline import explain_transaction


def process_transaction(row_df, deps):
    if isinstance(row_df, pd.DataFrame) == False:
        row_df = pd.DataFrame([row_df])

    ctx = {}
    ctx["row_df"] = row_df.copy()
    ctx["messages"] = []
    ctx["messages"].append({"from": "main", "to": "feature", "task": "start", "data": None})

    fe = transform_features(ctx["row_df"], deps["artf"])
    anom = run_anomaly_pipeline(fe, deps["model"], deps["feature_cols"], deps["pipeline_artf"])
    scored = aggregate_anomaly_scores(anom, train_artifacts=deps["pipeline_artf"])
    scored = apply_context_adjustments(scored, artf=deps["ctx_artf"])
    ctx["messages"].append({"from": "feature", "to": "rule", "task": "skor hazir", "data": None})

    result = evaluate_rules(scored, rules_path=deps["rules_path"])
    ctx["result_df"] = result
    row = result.iloc[0]

    if row["rule_decision"] == "PASS" and row.get("final_anomaly_score", 1) < 0.15:
        ctx["explanation"] = {"explanation": "dusuk risk"}
    else:
        use_llm = True
        if "use_llm" in deps:
            use_llm = deps["use_llm"]
        ctx["explanation"] = explain_transaction(row, use_llm=use_llm)

    ctx["messages"].append({"from": "rule", "to": "main", "task": "bitti", "data": None})
    return ctx
