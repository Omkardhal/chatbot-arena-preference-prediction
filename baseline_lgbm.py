"""
Fast baseline for LLM Classification Finetuning (Chatbot Arena preferences).
Hand-crafted text features + LightGBM, with swap augmentation and swap TTA.
Runs on CPU in a few minutes. Expect log loss roughly 1.00 - 1.02.

Kaggle: add the competition data, turn Internet off, run, and it writes submission.csv.
"""
import json
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

DATA = "/kaggle/input/competitions/llm-classification-finetuning"
SEED = 42

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")


def clean(x):
    """prompt / response columns are JSON lists of strings (entries can be null)."""
    try:
        return " ".join(s for s in json.loads(x) if s is not None)
    except Exception:
        return str(x)


for df in (train, test):
    for c in ("prompt", "response_a", "response_b"):
        df[c] = df[c].map(clean)

REFUSAL = re.compile(r"\b(sorry|i cannot|i can't|as an ai|i'm unable|i am unable|unable to)\b", re.I)


def resp_feats(t):
    words = t.split()
    n_words = len(words)
    return {
        "chars": len(t),
        "words": n_words,
        "lines": t.count("\n"),
        "bullets": len(re.findall(r"^\s*([-*•]|\d+\.)\s", t, flags=re.M)),
        "code": t.count("```"),
        "headers": len(re.findall(r"^#{1,6}\s", t, flags=re.M)),
        "bold": t.count("**"),
        "digits": sum(ch.isdigit() for ch in t),
        "questions": t.count("?"),
        "excl": t.count("!"),
        "sentences": len(re.findall(r"[.!?]+", t)),
        "avg_word": np.mean([len(w) for w in words]) if n_words else 0.0,
        "uniq_ratio": len(set(words)) / n_words if n_words else 0.0,
        "refusal": int(bool(REFUSAL.search(t))),
    }


def build(prompt, ra, rb):
    fa, fb = resp_feats(ra), resp_feats(rb)
    row = {"prompt_chars": len(prompt), "prompt_words": len(prompt.split())}
    for k in fa:
        row[f"a_{k}"] = fa[k]
        row[f"b_{k}"] = fb[k]
        row[f"d_{k}"] = fa[k] - fb[k]
    row["len_ratio"] = np.log1p(fa["chars"]) - np.log1p(fb["chars"])
    row["same_resp"] = int(ra.strip() == rb.strip())
    return row


def make_X(df, swap=False):
    rows = [
        build(p, b, a) if swap else build(p, a, b)
        for p, a, b in zip(df.prompt, df.response_a, df.response_b)
    ]
    return pd.DataFrame(rows)


# labels: 0 = a wins, 1 = b wins, 2 = tie
y = train[["winner_model_a", "winner_model_b", "winner_tie"]].values.argmax(1)
y_swap = np.where(y == 0, 1, np.where(y == 1, 0, 2))

X, X_sw = make_X(train), make_X(train, swap=True)
T, T_sw = make_X(test), make_X(test, swap=True)

params = dict(
    objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=31,
    min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbosity=-1, seed=SEED,
)

oof = np.zeros((len(train), 3))
test_pred = np.zeros((len(test), 3))
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

for fold, (tr, va) in enumerate(skf.split(X, y)):
    # swap augmentation: add the a<->b flipped copy of the training rows
    Xtr = pd.concat([X.iloc[tr], X_sw.iloc[tr]])
    ytr = np.concatenate([y[tr], y_swap[tr]])
    dtr = lgb.Dataset(Xtr, ytr)
    dva = lgb.Dataset(X.iloc[va], y[va])
    model = lgb.train(
        params, dtr, num_boost_round=3000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    # validation prediction with swap TTA
    p = model.predict(X.iloc[va])
    p_sw = model.predict(X_sw.iloc[va])[:, [1, 0, 2]]
    oof[va] = (p + p_sw) / 2
    # test prediction with swap TTA
    t = model.predict(T)
    t_sw = model.predict(T_sw)[:, [1, 0, 2]]
    test_pred += (t + t_sw) / 2 / skf.n_splits
    print(f"fold {fold}: log loss = {log_loss(y[va], oof[va]):.4f} (iters={model.best_iteration})")

print(f"CV log loss: {log_loss(y, oof):.4f}")

sub = pd.DataFrame({
    "id": test["id"],
    "winner_model_a": test_pred[:, 0],
    "winner_model_b": test_pred[:, 1],
    "winner_tie": test_pred[:, 2],
})
sub.to_csv("submission.csv", index=False)
print(sub.head())
