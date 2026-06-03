# -*- coding: utf-8 -*-
"""
@author: mehme
"""

import os
import pickle

import numpy as np
import requests

POLICY_DIR = os.path.join(os.path.dirname(__file__), "..", "config", "fraud_policies")
INDEX_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "policy_index.pkl")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

embed_model = None
policy_index = None


def get_embed_model():
    global embed_model
    if embed_model is None:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embed_model


def load_policy_index(policies_dir=None):
    return build_index(policies_dir)


def build_index(policies_dir=None):
    global policy_index
    if policy_index is not None:
        return policy_index

    if policies_dir is None:
        policies_dir = POLICY_DIR

    if os.path.exists(INDEX_FILE):
        f = open(INDEX_FILE, "rb")
        policy_index = pickle.load(f)
        f.close()
        return policy_index

    chunks = []
    files = os.listdir(policies_dir)
    files.sort()
    for fn in files:
        if fn.endswith(".txt") == False:
            continue
        path = os.path.join(policies_dir, fn)
        f = open(path, encoding="utf-8")
        lines = f.readlines()
        f.close()
        for line in lines:
            t = line.strip()
            if t != "":
                chunks.append({"source": fn, "text": t})

    texts = []
    for c in chunks:
        texts.append(c["text"])

    model = get_embed_model()
    emb = model.encode(texts, normalize_embeddings=True)
    policy_index = {"chunks": chunks, "embeddings": np.array(emb)}

    f = open(INDEX_FILE, "wb")
    pickle.dump(policy_index, f)
    f.close()
    return policy_index


def search_policies(query, top_k=3):
    idx = build_index()
    model = get_embed_model()
    q = model.encode([query], normalize_embeddings=True)[0]

    sims = []
    for i in range(len(idx["embeddings"])):
        sim = float(np.dot(idx["embeddings"][i], q))
        sims.append(sim)

    order = np.argsort(sims)[::-1]
    hits = []
    for i in range(top_k):
        if i >= len(order):
            break
        j = order[i]
        hit = idx["chunks"][j].copy()
        hit["score"] = sims[j]
        hits.append(hit)
    return hits


def make_query(row):
    parts = []
    parts.append("Tutar " + str(row.get("TransactionAmt")) + " USD")
    parts.append("Saat " + str(row.get("Transaction_Hour")))
    skor = row.get("rule_final_score")
    if skor is None:
        skor = row.get("final_anomaly_score")
    parts.append("Skor " + str(skor))
    parts.append("Karar " + str(row.get("rule_decision")))
    exp = row.get("rule_explanations")
    if exp:
        parts.append(str(exp))

    q = ""
    for p in parts:
        if p and p != "None":
            q = q + p + " "
    return q.strip()


def ask_ollama(text):
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": text, "stream": False},
            timeout=60,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("response", "").strip()
    except:
        return None 


def rag_reason(row, query=None, top_k=3, use_llm=True):
    if query is None:
        query = make_query(row)

    hits = search_policies(query, top_k)

    policy_txt = ""
    for h in hits:
        policy_txt = policy_txt + "[" + h["source"] + "] " + h["text"] + "\n"

    skor = row.get("rule_final_score")
    if skor is None:
        skor = row.get("final_anomaly_score")

    context = "tutar=" + str(row.get("TransactionAmt"))
    context = context + " saat=" + str(row.get("Transaction_Hour"))
    context = context + "\nkarar=" + str(row.get("rule_decision"))
    context = context + " skor=" + str(skor)
    context = context + "\nkurallar=" + str(row.get("rule_explanations", "-"))
    context = context + "\npolicy:\n" + policy_txt

    answer = None
    if use_llm:
        answer = ask_ollama("fraud acikla:\n" + context)

    if answer is None or answer == "":
        if len(hits) > 0:
            top = hits[0]["text"]
        else:
            top = "-"
        answer = "Karar " + str(row.get("rule_decision"))
        answer = answer + ": " + str(row.get("rule_explanations", "-"))
        answer = answer + ". policy: " + top

    return {
        "query": query,
        "policy_snippets": hits,
        "context": context,
        "explanation": answer,
    }


def explain_transaction(row, query=None, top_k=3, use_llm=True):
    return rag_reason(row, query, top_k, use_llm)
