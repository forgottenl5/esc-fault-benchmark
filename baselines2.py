"""v2 reference baselines and the sample-size study.

usage:
  python baselines2.py --data data_v2 --tasks tasks_v2 --eval val                 # score on validation
  python baselines2.py --data data_v2 --tasks tasks_v2 --eval test,test_shift --no-score   # predictions only
  python baselines2.py ... --curve                                                   # also run the learning curves

Model: one gradient-boosted tree classifier per fault (multi-label), on the causal features in
escbench/tasks_v2.py. Task 2 streams the same model over each chuck; task 3 uses a gradient-boosted regressor.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from escbench.sim2 import FAULTS
from escbench.tasks_v2 import feature_matrix, normalise, window_features, to_multihot, from_multihot, REF_WAFERS
import score2


def clf():
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, class_weight="balanced", random_state=0)


class MultiFault:
    def fit(self, X, Y):
        self.m = [clf().fit(X, Y[:, k]) for k in range(Y.shape[1])]
        return self

    def proba(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.m])

    def predict(self, X, thr=0.5):
        return (self.proba(X) >= thr).astype(int)


def cached_features(a, split, kind="task1"):
    path = os.path.join(a.out, f"X_{kind}_{split}.npy")
    if os.path.exists(path):
        return np.load(path)
    w = pd.read_parquet(os.path.join(a.data, f"wafers_{split}.parquet"))
    rows = pd.read_csv(os.path.join(a.tasks, f"{kind}_{split}.csv"))
    X = feature_matrix(w, rows)
    np.save(path, X)
    return X


def stream_alarms(model, w, chucks, step=10, need=3, thr=0.6):
    groups = {c: g.sort_values("wafer_idx").reset_index(drop=True) for c, g in w.groupby("chuck_id")}
    out = []
    for cid in chucks:
        g = groups[cid]; nm = normalise(g)
        ends = list(range(REF_WAFERS + 29, len(g), step))
        P = model.proba(np.array([window_features(g, nm, e) for e in ends], dtype=float))
        streak, last, alarm = 0, None, (-1, "")
        for e, p in zip(ends, P):
            k = int(p.argmax())
            if p[k] >= thr:
                streak = streak + 1 if k == last else 1; last = k
                if streak >= need:
                    alarm = (e, FAULTS[k]); break
            else:
                streak, last = 0, None
        out.append((cid, alarm[0], alarm[1]))
    return pd.DataFrame(out, columns=["chuck_id", "alarm_wafer", "fault"])


def rul_rows(a):
    meta = json.load(open(os.path.join(a.data, "chucks_train.json")))
    rows = []
    for m in meta:
        if m["run_to_failure"]:
            f = [x for x in m["faults"] if x in ("contamination", "wear")][0]
            for e in range(m["onsets"][f], m["n_wafers"] - 1, 10):
                rows.append((m["chuck_id"], e, m["n_wafers"] - 1 - e))
    return pd.DataFrame(rows, columns=["chuck_id", "end_wafer", "rul"])


def learning_curve(a, Xtr, ttr, evals):
    meta = {m["chuck_id"]: m for m in json.load(open(os.path.join(a.data, "chucks_train.json")))}
    key = {c: "+".join(sorted(m["faults"])) or "healthy" for c, m in meta.items()}
    by = {}
    for c, k in key.items():
        by.setdefault(k, []).append(c)
    Ytr = to_multihot(ttr.label.values)
    res = []
    for per in [5, 10, 20, 40]:
        for rep in range(3 if per < 40 else 1):
            rng = np.random.default_rng([per, rep])
            keep = []
            for k, cs in by.items():
                n = per if "+" not in k else max(1, round(5 * per / 40))
                keep += list(rng.choice(cs, size=min(n, len(cs)), replace=False))
            sel = ttr.chuck_id.isin(keep).values
            m = MultiFault().fit(Xtr[sel], Ytr[sel])
            for split, (X, ans) in evals.items():
                pred = ans[["chuck_id", "end_wafer"]].assign(label=from_multihot(m.predict(X)))
                s = score2.score_task1(pred, ans)
                res.append(dict(per_class=per, rep=rep, split=split, macro_f1=s["macro_f1"], **s["per_fault_f1"]))
                print(per, rep, split, round(s["macro_f1"], 3), s["per_fault_f1"], flush=True)
    return pd.DataFrame(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v2"); ap.add_argument("--tasks", default="tasks_v2")
    ap.add_argument("--eval", default="val"); ap.add_argument("--out", default="baselines_out_v2")
    ap.add_argument("--no-score", action="store_true"); ap.add_argument("--answers-dir", default=None)
    ap.add_argument("--curve", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ttr = pd.read_csv(os.path.join(a.tasks, "task1_train.csv"))
    Xtr = cached_features(a, "train")
    model = MultiFault().fit(Xtr, to_multihot(ttr.label.values))
    rr = rul_rows(a)
    wtr = pd.read_parquet(os.path.join(a.data, "wafers_train.parquet"))
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(feature_matrix(wtr, rr), rr.rul)
    scores, evals = {}, {}
    for split in a.eval.split(","):
        rows1 = pd.read_csv(os.path.join(a.tasks, f"task1_{split}.csv"))
        X1 = cached_features(a, split)
        p1 = rows1[["chuck_id", "end_wafer"]].assign(label=from_multihot(model.predict(X1)))
        w = pd.read_parquet(os.path.join(a.data, f"wafers_{split}.parquet"))
        rows2 = pd.read_csv(os.path.join(a.tasks, f"task2_{split}.csv"))
        p2 = stream_alarms(model, w, rows2.chuck_id.tolist())
        rows3 = pd.read_csv(os.path.join(a.tasks, f"task3_{split}.csv"))
        p3 = rows3[["chuck_id", "end_wafer"]].assign(rul_pred=reg.predict(feature_matrix(w, rows3)).round(1))
        for k, p in ((1, p1), (2, p2), (3, p3)):
            p.to_csv(os.path.join(a.out, f"{split}_task{k}_gbt.csv"), index=False)
        if a.no_score:
            continue
        adir = a.answers_dir if split in ("test", "test_shift") else None
        ans = [pd.read_csv(os.path.join(adir, f"task{k}_{split}_answers.csv")) if adir else rows for k, rows in
               ((1, rows1), (2, rows2), (3, rows3))]
        ans[2] = ans[2][ans[2].rul >= 0]
        scores[split] = {"task1": score2.score_task1(p1, ans[0]), "task2": score2.score_task2(p2, ans[1]),
                         "task3": score2.score_task3(p3, ans[2])}
        evals[split] = (X1, ans[0])
        print(split, json.dumps(scores[split], indent=1), flush=True)
    json.dump(scores, open(os.path.join(a.out, "scores.json"), "w"), indent=1)
    if a.curve and evals:
        lc = learning_curve(a, Xtr, ttr, evals)
        lc.to_csv(os.path.join(a.out, "learning_curve.csv"), index=False)


if __name__ == "__main__":
    main()
