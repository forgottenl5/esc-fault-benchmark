"""Reference baselines for all three tasks.

usage: python baselines.py --data data --tasks tasks --eval val        (scores against validation answers)
       python baselines.py --data data --tasks tasks --eval test --no-score   (writes test predictions only)

Models:
  rules  - fixed thresholds on drift from each chuck's first 100 wafers (like the patents' alarms)
  rf     - random forest on the same features
  mlp    - small neural network (sklearn MLP) on the same features
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from escbench.tasks import FEATURE_NAMES, feature_matrix, window_features, REF_WAFERS
import score

F = {n: i for i, n in enumerate(FEATURE_NAMES)}


def rules(X):
    out = []
    for x in X:
        leak, flow, pset = x[F["r_i_leak_mean"]], x[F["r_he_flow_mean"]], x[F["r_he_p_set_mean"]]
        if x[F["d_dechuck_med"]] > 0.6 or x[F["stick_rate"]] > 0.15:
            out.append("residual_charge")
        elif x[F["zone_dev_max"]] > 1.2:
            # a real heater fault moves the controller's helium setpoint; a bad sensor does not
            out.append("heater_zone" if abs(pset - 1) > 0.03 else "sensor_drift")
        elif abs(x[F["d_p_gap"]]) > 0.06:
            out.append("sensor_drift")
        elif leak > 1.35:
            out.append("contamination" if flow > 1.1 else "sensor_drift")
        elif flow > 1.35:
            out.append("he_leak" if x[F["d_ch_press_mean"]] > 0.25 else "sensor_drift")
        elif pset < 0.85:
            out.append("wear")
        else:
            out.append("healthy")
    return np.array(out)


def load(a, split):
    w = pd.read_parquet(os.path.join(a.data, f"wafers_{split}.parquet"))
    t = {k: pd.read_csv(os.path.join(a.tasks, f"task{k}_{split}.csv")) for k in (1, 2, 3)}
    return w, t


def train_rows_rul(w, lab_meta):
    rows = []
    for m in lab_meta:
        if not m["run_to_failure"]:
            continue
        N, on = m["n_wafers"], m["onset_wafer"]
        for e in range(on, N - 1, 10):
            rows.append((m["chuck_id"], e, N - 1 - e))
    return pd.DataFrame(rows, columns=["chuck_id", "end_wafer", "rul"])


def stream_alarms(model, w, chucks, step=10, need=3, thr=0.6):
    groups = {cid: g.sort_values("wafer_idx").reset_index(drop=True) for cid, g in w.groupby("chuck_id")}
    classes = list(model.classes_) if hasattr(model, "classes_") else None
    rows = []
    for cid in chucks:
        g = groups[cid]; streak = 0; last = None; alarm = (-1, "")
        ends = list(range(REF_WAFERS + 29, len(g), step))
        X = np.array([window_features(g, e) for e in ends])
        if classes is None:
            labs = rules(X); conf = np.ones(len(labs))
        else:
            p = model.predict_proba(X); labs = np.array(classes)[p.argmax(1)]; conf = p.max(1)
        for e, lab, c in zip(ends, labs, conf):
            if lab != "healthy" and c >= thr:
                streak = streak + 1 if lab == last else 1; last = lab
                if streak >= need:
                    alarm = (e, lab); break
            else:
                streak = 0; last = None
        rows.append((cid, alarm[0], alarm[1]))
    return pd.DataFrame(rows, columns=["chuck_id", "alarm_wafer", "fault"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--tasks", default="tasks")
    ap.add_argument("--eval", default="val"); ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--out", default="baselines_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    wtr, ttr = load(a, "train"); wev, tev = load(a, a.eval)
    meta_tr = json.load(open(os.path.join(a.data, "chucks_train.json")))

    Xtr = feature_matrix(wtr, ttr[1]); ytr = ttr[1].label.values
    Xev = feature_matrix(wev, tev[1])
    models = {
        "rules": None,
        "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(Xtr, ytr),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=800, random_state=0)).fit(Xtr, ytr),
    }
    # task 3 regressor
    rul_rows = train_rows_rul(wtr, meta_tr)
    reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(
        feature_matrix(wtr, rul_rows), rul_rows.rul.values)

    results = {}
    for name, mdl in models.items():
        p1 = rules(Xev) if mdl is None else mdl.predict(Xev)
        d1 = tev[1][["chuck_id", "end_wafer"]].assign(label=p1)
        d2 = stream_alarms(mdl, wev, tev[2].chuck_id.tolist())
        d1.to_csv(f"{a.out}/{a.eval}_task1_{name}.csv", index=False)
        d2.to_csv(f"{a.out}/{a.eval}_task2_{name}.csv", index=False)
        if not a.no_score:
            results[name] = {"task1": score.score_task1(d1, tev[1]), "task2": score.score_task2(d2, tev[2])}
    p3 = reg.predict(feature_matrix(wev, tev[3])) if len(tev[3]) else []
    d3 = tev[3][["chuck_id", "end_wafer"]].assign(rul_pred=np.round(p3))
    d3.to_csv(f"{a.out}/{a.eval}_task3_rf.csv", index=False)
    # naive task-3 reference: predict the training-set mean remaining life
    d3n = d3.assign(rul_pred=float(rul_rows.rul.mean()))
    if not a.no_score:
        results["rf"]["task3"] = score.score_task3(d3, tev[3])
        results["mean_guess"] = {"task3": score.score_task3(d3n, tev[3])}
        print(json.dumps(results, indent=1))
        json.dump(results, open(f"{a.out}/{a.eval}_scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
