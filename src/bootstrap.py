# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import lightgbm as lgb

from data_load import load_and_merge_data
from validation import split_data_chronologically, verify_split_integrity
from features import extract_feature_artifacts, transform_features, build_model_matrix
from anomaly_engine import run_anomaly_pipeline
from scores import aggregate_anomaly_scores
from context_adjust import extract_context_artifacts, apply_context_adjustments
from rule_engine import evaluate_rules
from rag_pipeline import load_policy_index


def score_splits(fe_map, model, cols, ctx_artf, rules_path):
    pipe_artf = {}
    out = {}

    for name in fe_map:
        fe = fe_map[name]
        anom = run_anomaly_pipeline(fe, model, cols, pipe_artf)
        df = aggregate_anomaly_scores(anom, train_artifacts=pipe_artf)
        df = apply_context_adjustments(df, artf=ctx_artf)
        df = evaluate_rules(df, rules_path=rules_path)
        out[name] = df

    return out, pipe_artf


def init_pipeline(data_dir, rules_path, policies_dir):
    print("pipeline yukleniyor, biraz bekle...")

    df = load_and_merge_data(data_dir)
    train, val, test = split_data_chronologically(df)
    verify_split_integrity(train, val, test)

    x_train = train.drop(columns=["isFraud"])
    y_train = train["isFraud"]
    artf = extract_feature_artifacts(x_train, y_train)

    fe_train = transform_features(train, artf)
    fe_val = transform_features(val, artf)
    fe_test = transform_features(test, artf)
    fe_map = {"train": fe_train, "val": fe_val, "test": fe_test}

    y_tr = train["isFraud"]
    y_va = val["isFraud"]
    rate = y_tr.mean()
    print("lgbm egitiliyor...")

    model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.008,

    num_leaves=128,          

    min_child_samples=100,   
    min_child_weight=1e-3,

    subsample=0.8,          
    colsample_bytree=0.8,   

    reg_alpha=0.5,          
    reg_lambda=1.0,         

    scale_pos_weight=(1 - rate) / rate,  


    random_state=42,
    n_jobs=-1,
    verbose=-1
    )

    x_tr = build_model_matrix(fe_map["train"], artf)
    x_va = build_model_matrix(fe_map["val"], artf)
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_va, y_va)],
        eval_metric="average_precision",
        categorical_feature=artf["model_categorical_cols"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    ctx_artf = extract_context_artifacts(fe_map["train"])
    finals, pipe_artf = score_splits(fe_map, model, artf["model_feature_cols"], ctx_artf, rules_path)
    load_policy_index(policies_dir)

    deps = {}
    deps["artf"] = artf
    deps["model"] = model
    deps["feature_cols"] = artf["model_feature_cols"]
    deps["pipeline_artf"] = pipe_artf
    deps["ctx_artf"] = ctx_artf
    deps["rules_path"] = rules_path
    deps["use_llm"] = True

    return deps, val, finals["val"], y_va, fe_map, finals["test"], test["isFraud"]
