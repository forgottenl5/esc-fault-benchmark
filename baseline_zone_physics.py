"""Can heater physics separate a drifting zone-temperature sensor from a real heater-zone fault? (v2, task 1). Result: no gain on test; see README.

If one heater zone really runs hot or cold by d degrees, the chuck body follows (by about d/4 with four zones), so
the wafer temperature moves and the leakage current changes with it (dielectric resistance falls with temperature).
If only the zone *sensor* drifts, the reading moves by d but the wafer temperature and leakage do not.
This script adds a few window features built on that check to the summary features of escbench/tasks_v2.py:

  out_dev    signed deviation of the outlier zone from the other three (last 30 wafers)
  out_raw    the outlier zone's own deviation from its reference
  tw_dev     wafer-temperature deviation from reference
  explained  4 * tw_dev / out_raw: ~1 for a real heater change, ~0 for a lying sensor
  resid      tw_dev - out_raw / 4: ~0 for a real heater change, ~-d/4 for a lying sensor
  leak_t     log leakage ratio / out_raw (leakage follows a real temperature change only)
plus the change of each over the 30 wafers before.

Development uses 5-fold cross-validation over training chucks (grouped by chuck); test and test_shift are scored once.

usage: python baseline_zone_physics.py --cv                     # cross-validation on train only
       python baseline_zone_physics.py --eval val,test,test_shift --labels-dir data_v2/private_test_labels
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from escbench.tasks_v2 import normalise, to_multihot, from_multihot, REF_WAFERS
import score2

ZK = ["dz1", "dz2", "dz3", "dz4"]
NAMES = ["out_dev", "out_raw", "tw_dev", "explained", "resid", "leak_t"]
NAMES2 = ["pset_resid", "couple_slope", "couple_r", "couple_slope_resid"]
K_PSET = None      # helium set-point response per degree of real zone shift, fitted on training heater windows


def _window(nm, a, b):
    seg = nm.iloc[a:b]
    if len(seg) == 0 or seg[ZK].isna().all().all():
        return [np.nan] * len(NAMES)
    dz = np.nanmedian(seg[ZK].values, axis=0)
    k = int(np.argmax(np.abs(dz - (dz.sum() - dz) / 3)))
    out_dev = dz[k] - (dz.sum() - dz[k]) / 3
    out_raw = dz[k]
    tw = float(np.nanmedian(seg["t_wafer_mean"].values))
    lk = float(np.nanmedian(np.log(seg["i_leak_mean"].values)))
    den = out_raw if abs(out_raw) > 0.3 else np.sign(out_raw or 1) * 0.3
    return [out_dev, out_raw, tw, float(np.clip(4 * tw / den, -3, 3)), tw - out_raw / 4, lk / den]


def physics_features(w, nm, end):
    lo = max(REF_WAFERS, end - 29)
    now = _window(nm, lo, end + 1)
    prev = _window(nm, max(REF_WAFERS, end - 59), lo) if end - 59 >= REF_WAFERS else now
    return now + [a - b for a, b in zip(now, prev)]


def coupling(nm, end):
    """Over the last 90 wafers: does the helium set-point follow the outlier zone reading?
    A real heater change is compensated by the run-to-run controller; a lying sensor is not."""
    seg = nm.iloc[max(REF_WAFERS, end - 89):end + 1]
    cur = nm.iloc[max(REF_WAFERS, end - 29):end + 1]
    if len(seg) < 20 or cur[ZK].isna().all().all():
        return [np.nan] * 3
    dz = np.nanmedian(cur[ZK].values, axis=0)
    k = int(np.argmax(np.abs(dz - (dz.sum() - dz) / 3)))
    x = seg[ZK[k]].values; y = seg["he_p_set_mean"].values - 1
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 20 or x.std() < 1e-6:
        return [0.0, 0.0, float(np.nanmedian(y)) if len(y) else np.nan]
    b = float(np.polyfit(x, y, 1)[0])
    r = float(np.corrcoef(x, y)[0, 1]) if y.std() > 1e-9 else 0.0
    return [b, r, float(np.median(y[-30:]))]


def matrix2(wafers, rows):
    groups = {c: g.sort_values("wafer_idx").reset_index(drop=True) for c, g in wafers.groupby("chuck_id")}
    norms = {c: normalise(g) for c, g in groups.items()}
    return np.array([coupling(norms[r.chuck_id], int(r.end_wafer)) for r in rows.itertuples()], dtype=float)


def add_physics2(Z, C, k):
    """pset_resid: set-point change not explained by the zone reading; couple_slope_resid: slope minus k."""
    pset_now, out_raw = C[:, 2], Z[:, 1]
    return np.column_stack([pset_now - k * out_raw, C[:, 0], C[:, 1], C[:, 0] - k])


def matrix(wafers, rows):
    groups = {c: g.sort_values("wafer_idx").reset_index(drop=True) for c, g in wafers.groupby("chuck_id")}
    norms = {c: normalise(g) for c, g in groups.items()}
    return np.array([physics_features(groups[r.chuck_id], norms[r.chuck_id], int(r.end_wafer)) for r in rows.itertuples()],
                    dtype=float)


def cached(a, split, rows, kind="Z"):
    p = os.path.join(a.out, f"{kind}_task1_{split}.npy")
    if not os.path.exists(p):
        fn = matrix if kind == "Z" else matrix2
        np.save(p, fn(pd.read_parquet(os.path.join(a.data, f"wafers_{split}.parquet")), rows))
    return np.load(p)


def fit_k(Z, C, labels):
    """Least-squares set-point response per degree, from windows whose only fault is heater_zone."""
    m = (labels == "heater_zone") & np.isfinite(Z[:, 1]) & np.isfinite(C[:, 2]) & (np.abs(Z[:, 1]) > 0.5)
    x, y = Z[m, 1], C[m, 2]
    return float((x * y).sum() / (x * x).sum())


def fit(X, Y):
    return [HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, class_weight="balanced", random_state=0)
            .fit(X, Y[:, k]) for k in range(Y.shape[1])]


def predict(ms, X):
    return np.column_stack([m.predict_proba(X)[:, 1] for m in ms]) >= 0.5


def zone_recall(rows, pred, meta):
    d = rows.assign(pred=from_multihot(pred))
    d = d[d.label.str.contains("sensor_drift")]
    z = d.chuck_id.map(lambda c: meta[c]["drift_sensor"].startswith("t_z"))
    d = d[z.values]
    hz = rows.assign(pred=from_multihot(pred))
    hz = hz[hz.label.str.contains("heater_zone")]
    return dict(zone_drift_windows=int(len(d)), zone_drift_chucks=int(d.chuck_id.nunique()),
                zone_drift_recall=round(float(d.pred.str.contains("sensor_drift").mean()), 3) if len(d) else None,
                zone_drift_called_heater=round(float(d.pred.str.contains("heater_zone").mean()), 3) if len(d) else None,
                heater_recall=round(float(hz.pred.str.contains("heater_zone").mean()), 3))


def report(tag, rows, pred, meta):
    s = score2.score_task1(rows[["chuck_id", "end_wafer"]].assign(label=from_multihot(pred)), rows)
    r = dict(macro_f1=round(s["macro_f1"], 3), sensor_drift_f1=s["per_fault_f1"]["sensor_drift"],
             heater_zone_f1=s["per_fault_f1"]["heater_zone"], **zone_recall(rows, pred, meta))
    print(tag, json.dumps(r), flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v2"); ap.add_argument("--tasks", default="tasks_v2")
    ap.add_argument("--out", default="baselines_out_v2"); ap.add_argument("--labels-dir", default=None)
    ap.add_argument("--cv", action="store_true"); ap.add_argument("--eval", default="")
    a = ap.parse_args()
    ttr = pd.read_csv(os.path.join(a.tasks, "task1_train.csv"))
    Str = np.load(os.path.join(a.out, "X_task1_train.npy"))
    Ztr = cached(a, "train", ttr); Ctr = cached(a, "train", ttr, "C")
    Ytr = to_multihot(ttr.label.values)
    meta_tr = {m["chuck_id"]: m for m in json.load(open(os.path.join(a.data, "chucks_train.json")))}
    variants = {"summary only": lambda S, Z, C, k: S,
                "summary + physics v1": lambda S, Z, C, k: np.hstack([S, Z]),
                "summary + physics v2": lambda S, Z, C, k: np.hstack([S, Z, add_physics2(Z, C, k)])}
    results = {}
    if a.cv:
        for name, mk in variants.items():
            pred = np.zeros_like(Ytr, dtype=bool)
            for tr, te in GroupKFold(5).split(Str, groups=ttr.chuck_id):
                k = fit_k(Ztr[tr], Ctr[tr], ttr.label.values[tr])           # fitted inside the fold
                X = mk(Str, Ztr, Ctr, k)
                pred[te] = predict(fit(X[tr], Ytr[tr]), X[te])
            results[f"cv | {name}"] = report(f"cv | {name}", ttr, pred, meta_tr)
    if a.eval:
        k = fit_k(Ztr, Ctr, ttr.label.values); print("k =", round(k, 4))
        models = {n: fit(mk(Str, Ztr, Ctr, k), Ytr) for n, mk in variants.items()}
        for split in a.eval.split(","):
            private = split in ("test", "test_shift")
            rows = pd.read_csv(os.path.join(a.tasks, f"task1_{split}.csv"))
            ldir = a.labels_dir if private else a.data
            if private:
                rows = rows.merge(pd.read_csv(os.path.join(ldir, f"task1_{split}_answers.csv")), on=["chuck_id", "end_wafer"])
            meta = {m["chuck_id"]: m for m in json.load(open(os.path.join(ldir, f"chucks_{split}.json")))}
            S = np.load(os.path.join(a.out, f"X_task1_{split}.npy")); Z = cached(a, split, rows); C = cached(a, split, rows, "C")
            for name, mk in variants.items():
                results[f"{split} | {name}"] = report(f"{split} | {name}", rows, predict(models[name], mk(S, Z, C, k)), meta)
    json.dump(results, open(os.path.join(a.out, "zone_physics_scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
