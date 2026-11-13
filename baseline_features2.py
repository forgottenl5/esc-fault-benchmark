"""Task 1 baseline with extended features (trends, step size, cross-signal consistency).

usage: python baseline_features2.py --data data --tasks tasks --eval val
       python baseline_features2.py --data data --tasks tasks --eval test --no-score    (writes predictions only)
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from escbench.features2 import feature_matrix2
import score


def load(a, split):
    return pd.read_parquet(os.path.join(a.data, f"wafers_{split}.parquet")), pd.read_csv(os.path.join(a.tasks, f"task1_{split}.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--tasks", default="tasks")
    ap.add_argument("--eval", default="val"); ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--out", default="baselines_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wt, tt = load(a, "train")
    clf = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1)
    clf.fit(feature_matrix2(wt, tt), tt.label.values)
    we, te = load(a, a.eval)
    pred = te[["chuck_id", "end_wafer"]].assign(label=clf.predict(feature_matrix2(we, te)))
    path = os.path.join(a.out, f"{a.eval}_task1_rf_features2.csv"); pred.to_csv(path, index=False)
    print("wrote", path)
    if not a.no_score:
        print(score.score_task1(pred, te))


if __name__ == "__main__":
    main()
