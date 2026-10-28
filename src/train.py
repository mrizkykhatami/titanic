"""Training + Cross-Validation (CV) + submission untuk Titanic.

Jalankan dari folder repo:
    python -m src.train                          # pakai configs/baseline.yaml
    python -m src.train --config configs/exp002.yaml
    python -m src.train --model rf --seed 7      # override cepat

Hasil per eksperimen ada di outputs/<nama>/ :
    oof.csv          prediksi out-of-fold (untuk analisis & ensemble)
    submission.csv   file yang dikirim ke Kaggle
    metrics.json     skor CV per fold
Ringkasan semua eksperimen ada di outputs/exp_log.csv.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import load
from .env import ENV, OUTPUT_DIR, ROOT
from .features import CAT_COLS, NUM_COLS, add_features

MODELS = {
    "logreg": lambda seed: LogisticRegression(max_iter=2000, C=0.5),
    "rf": lambda seed: RandomForestClassifier(
        n_estimators=500, max_depth=7, min_samples_leaf=2, random_state=seed, n_jobs=-1
    ),
    "hgb": lambda seed: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=seed
    ),
}


def build_pipeline(model: str, seed: int) -> Pipeline:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])
    return Pipeline([("pre", pre), ("model", MODELS[model](seed))])


def run_cv(model: str = "hgb", n_splits: int = 5, seed: int = 42, name: str | None = None) -> dict:
    """K-fold CV: tiap fold, latih di (K-1) bagian, tebak bagian sisanya (OOF)."""
    t0 = time.time()
    train, test = load()
    train, test = add_features(train), add_features(test)
    X, y = train[NUM_COLS + CAT_COLS], train["Survived"].to_numpy()
    X_test = test[NUM_COLS + CAT_COLS]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_prob = np.zeros(len(X_test))
    fold_scores = []

    for k, (tr, va) in enumerate(skf.split(X, y)):
        pipe = build_pipeline(model, seed)
        pipe.fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        score = accuracy_score(y[va], oof[va] > 0.5)
        fold_scores.append(score)
        test_prob += pipe.predict_proba(X_test)[:, 1] / n_splits  # rata-rata K model
        print(f"  fold {k}: accuracy = {score:.4f}")

    cv = accuracy_score(y, oof > 0.5)
    print(f"CV accuracy (OOF) = {cv:.4f} | per-fold {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    name = name or f"{model}_s{seed}"
    out = OUTPUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PassengerId": train["PassengerId"], "oof_prob": oof, "Survived": y}).to_csv(
        out / "oof.csv", index=False
    )
    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": (test_prob > 0.5).astype(int)})
    sub.to_csv(out / "submission.csv", index=False)
    if ENV == "kaggle":  # supaya bisa langsung di-submit dari output notebook Kaggle
        sub.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    result = {
        "name": name, "model": model, "n_splits": n_splits, "seed": seed, "env": ENV,
        "cv_accuracy": round(float(cv), 5),
        "fold_mean": round(float(np.mean(fold_scores)), 5),
        "fold_std": round(float(np.std(fold_scores)), 5),
        "fold_scores": [round(float(s), 5) for s in fold_scores],
        "seconds": round(time.time() - t0, 1),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    log = OUTPUT_DIR / "exp_log.csv"
    row = {k: v for k, v in result.items() if k != "fold_scores"}
    pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
    print(f"[saved] {out}")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Titanic CV training")
    p.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    p.add_argument("--model", choices=sorted(MODELS))
    p.add_argument("--n-splits", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--name")
    a = p.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text()) if a.config else {}
    for key in ("model", "n_splits", "seed", "name"):
        val = getattr(a, key)
        if val is not None:
            cfg[key] = val
    if a.model and not a.name:  # override model tanpa nama -> nama otomatis
        cfg.pop("name", None)
    run_cv(cfg.get("model", "hgb"), cfg.get("n_splits", 5), cfg.get("seed", 42), cfg.get("name"))


if __name__ == "__main__":
    main()
