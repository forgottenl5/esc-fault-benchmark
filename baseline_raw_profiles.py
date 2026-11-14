"""Does the raw 1 Hz signal help? Task-1 baseline on within-wafer signal shapes (v2).

For every wafer, each signal's 90-second trace is averaged into 30 three-second blocks and compared with the
chuck's own reference trace for the same recipe (median of its first 100 wafers). A window's profile is the
median deviation over its last 30 wafers, plus its change from the 30 wafers before. One gradient-boosted
classifier per fault is trained on (a) these profiles alone and (b) profiles plus the summary features of
escbench/tasks_v2.py, and compared with the summary-only baseline from baselines2.py.

Raw signals are rebuilt chuck by chuck from the generator (identical to the published files), so nothing large is
kept in memory. Test splits need the private seed (maintainers); the public run uses val.

usage: python baseline_raw_profiles.py --eval val
       python baseline_raw_profiles.py --eval test,test_shift --labels-dir data_v2/private_test_labels
"""
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from escbench.sim2 import simulate_chuck, CHANNELS, FAULTS
from escbench.features_v2 import quantize
from escbench.tasks_v2 import to_multihot, from_multihot, REF_WAFERS
from generate2 import plan_split, ID_BASE, PUBLIC_SEED
import score2

USE = ["v_rb", "i_leak_ma", "he_p_set_torr", "he_p_torr", "he_flow_sccm", "t_wafer_c",
       "t_z1_c", "t_z2_c", "t_z3_c", "t_z4_c", "ch_press_mtorr"]
RELATIVE = {"v_rb", "i_leak_ma", "he_p_set_torr", "he_p_torr", "he_flow_sccm"}
IDX = [CHANNELS.index(c) for c in USE]
BLOCK = 3


def chuck_stream(split, seed):
    """Yield (chuck_id, quantized raw signals, recipe per wafer) exactly as generate2.py builds them."""
    tool = "B" if split == "test_shift" else "A"
    groups = plan_split(split)
    order = np.random.default_rng([seed, ID_BASE[split]]).permutation(len(groups))
    for j, gi in enumerate(order):
        cid = ID_BASE[split] + j
        rng = np.random.default_rng([seed, cid])
        new_at = int(rng.integers(300, 650)) if rng.random() < 0.3 else None
        r = simulate_chuck(rng, groups[gi], cid, tool=tool, new_recipe_at=new_at)
        yield cid, quantize(r["ts"]), r["wafer"]["recipe"]


def profiles(ts, recipe, ends):
    N = ts.shape[0]
    x = ts[:, :, IDX].reshape(N, 90 // BLOCK, BLOCK, len(IDX)).mean(axis=2)        # (N, 30, 11)
    dev = np.full_like(x, np.nan)
    for r in np.unique(recipe):
        ref_sel = np.nonzero(recipe[:REF_WAFERS] == r)[0]
        sel = recipe == r
        if len(ref_sel) == 0:
            continue                                                               # recipe with no reference
        ref = np.median(x[ref_sel], axis=0)                                        # (30, 11)
        level = np.abs(ref[5:26]).mean(axis=0) + 1e-6                              # steady-window level
        d = x[sel] - ref
        for k, c in enumerate(USE):
            if c in RELATIVE:
                d[:, :, k] /= level[k]
        dev[sel] = d
    out = []
    for e in ends:
        now = dev[max(REF_WAFERS, e - 29):e + 1]
        prev = dev[max(REF_WAFERS, e - 59):max(REF_WAFERS, e - 29)]
        pn = np.nanmedian(now, axis=0) if np.isfinite(now).any() else np.full(x.shape[1:], np.nan)
        pp = np.nanmedian(prev, axis=0) if len(prev) and np.isfinite(prev).any() else pn
        out.append(np.concatenate([pn.ravel(), (pn - pp).ravel()]))
    return np.array(out, dtype=np.float32)


def split_profiles(split, seed, tasks_dir, cache):
    path = os.path.join(cache, f"P_{split}.npy")
    rows = pd.read_csv(os.path.join(tasks_dir, f"task1_{split}.csv"))
    if os.path.exists(path):
        return np.load(path), rows
    ends = rows.groupby("chuck_id").end_wafer.apply(list).to_dict()
    pos = {(c, e): i for i, (c, e) in enumerate(zip(rows.chuck_id, rows.end_wafer))}
    P = None
    for cid, ts, recipe in chuck_stream(split, seed):
        if cid not in ends:
            continue
        pr = profiles(ts, recipe, ends[cid])
        if P is None:
            P = np.full((len(rows), pr.shape[1]), np.nan, dtype=np.float32)
        for e, v in zip(ends[cid], pr):
            P[pos[(cid, e)]] = v
    np.save(path, P)
    return P, rows


def fit(X, Y):
    return [HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, class_weight="balanced", random_state=0)
            .fit(X, Y[:, k]) for k in range(Y.shape[1])]


def predict(models, X):
    return np.column_stack([m.predict_proba(X)[:, 1] for m in models]) >= 0.5


def sensor_recall(split, labels_dir, rows, P):
    meta = {m["chuck_id"]: m for m in json.load(open(os.path.join(labels_dir, f"chucks_{split}.json")))}
    d = rows.assign(pred=from_multihot(P))
    d = d[d.label.str.contains("sensor_drift")]
    d["sensor"] = d.chuck_id.map(lambda c: meta[c]["drift_sensor"]).str.replace(r"t_z\d_c", "zone_temperature", regex=True)
    return d.groupby("sensor").apply(lambda g: round(float(g.pred.str.contains("sensor_drift").mean()), 2)).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="val"); ap.add_argument("--tasks", default="tasks_v2")
    ap.add_argument("--labels-dir", default=None); ap.add_argument("--data", default="data_v2")
    ap.add_argument("--summary-cache", default="baselines_out_v2"); ap.add_argument("--out", default="baselines_out_v2")
    ap.add_argument("--test-seed-file", default="private/test_seed_v2.txt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Ptr, ttr = split_profiles("train", PUBLIC_SEED, a.tasks, a.out)
    Str = np.load(os.path.join(a.summary_cache, "X_task1_train.npy"))
    Ytr = to_multihot(ttr.label.values)
    variants = {"summary only": (Str,), "profiles only": (Ptr,), "summary + profiles": (Str, Ptr)}
    models = {k: fit(np.hstack(v), Ytr) for k, v in variants.items()}
    results = {}
    for split in a.eval.split(","):
        private = split in ("test", "test_shift")
        seed = int(open(a.test_seed_file).read().strip()) if private else PUBLIC_SEED
        P, rows = split_profiles(split, seed, a.tasks, a.out)
        S = np.load(os.path.join(a.summary_cache, f"X_task1_{split}.npy"))
        ldir = a.labels_dir if private else a.data
        if private:
            rows = rows.merge(pd.read_csv(os.path.join(ldir, f"task1_{split}_answers.csv")), on=["chuck_id", "end_wafer"])
        feats = {"summary only": (S,), "profiles only": (P,), "summary + profiles": (S, P)}
        for k, m in models.items():
            pred = predict(m, np.hstack(feats[k]))
            s = score2.score_task1(rows[["chuck_id", "end_wafer"]].assign(label=from_multihot(pred)), rows)
            results[f"{split} | {k}"] = dict(macro_f1=round(s["macro_f1"], 3), exact=round(s["exact_match"], 3),
                                             per_fault=s["per_fault_f1"], sensor_recall=sensor_recall(split, ldir, rows, pred))
            print(split, "|", k, json.dumps(results[f"{split} | {k}"]), flush=True)
    json.dump(results, open(os.path.join(a.out, "raw_profile_scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
