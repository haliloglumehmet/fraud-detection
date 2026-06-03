# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import numpy as np
import pandas as pd
import yaml


def load_rules(path):
    f = open(path, encoding="utf-8")
    cfg = yaml.safe_load(f)
    f.close()
    rules = cfg.get("rules", [])
    decisions = cfg.get("decisions", {})
    return rules, decisions


def check_op(s, op, val):
    if op == "gt":
        return s > val
    if op == "gte":
        return s >= val
    if op == "lt":
        return s < val
    if op == "lte":
        return s <= val
    if op == "eq":
        return s == val
    if op == "in":
        return s.isin(val)
    if op == "between":
        return s.between(val[0], val[1])
    raise ValueError("op tanimsiz: " + str(op))


def rule_hits(df, rule):
    mask = pd.Series(True, index=df.index)
    conds = rule.get("all", [])
    for c in conds:
        col = c["col"]
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        mask = mask & check_op(df[col], c["op"], c["val"])
    return mask.fillna(False)


def sort_rules_by_priority(rules):
    # basit siralama, priority buyuk olan once
    out = list(rules)
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            pi = out[i].get("priority", 0)
            pj = out[j].get("priority", 0)
            if pj > pi:
                out[i], out[j] = out[j], out[i]
    return out


def evaluate_rules(df, rules_path=None, rules=None, score_col="final_anomaly_score"):
    if rules is None:
        rules, dec = load_rules(rules_path)
    else:
        dec = {"block": 0.80, "review": 0.50}

    out = df.copy()
    boost = pd.Series(0.0, index=out.index)
    ids = pd.Series("", index=out.index, dtype=object)
    text = pd.Series("", index=out.index, dtype=object)

    rules_sorted = sort_rules_by_priority(rules)
    for rule in rules_sorted:
        hit = rule_hits(out, rule)
        if hit.any() == False:
            continue
        adj = rule.get("action", {}).get("adjust", 0)
        desc = rule.get("description", rule["id"])
        boost.loc[hit] = boost.loc[hit] + adj
        ids.loc[hit] = ids.loc[hit] + rule["id"] + ";"
        text.loc[hit] = text.loc[hit] + desc + " | "

    out["rule_score_boost"] = boost
    base_score = out[score_col].astype(float)
    out["rule_final_score"] = np.clip(base_score + boost, 0, 1)
    out["matched_rules"] = ids.str.rstrip(";")
    out["rule_explanations"] = text.str.rstrip(" | ")

    block_lim = dec.get("block", 0.8)
    review_lim = dec.get("review", 0.5)
    out["rule_decision"] = "PASS"
    out.loc[out["rule_final_score"] >= review_lim, "rule_decision"] = "REVIEW"
    out.loc[out["rule_final_score"] >= block_lim, "rule_decision"] = "BLOCK"
    return out
