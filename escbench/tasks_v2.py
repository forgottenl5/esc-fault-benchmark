"""v2 task definitions and a reference causal feature extractor.

Features use only wafers 0..end of one chuck. Each signal is compared with the chuck's own first 100 wafers,
recipe by recipe; a recipe that never ran in those wafers (the mid-life new recipe) gives missing values (NaN),
which the reference models handle natively.
"""
import numpy as np
import pandas as pd
from .sim2 import FAULTS

REF_WAFERS = 100
SEV_LABEL = 0.15
RUL_FRACS = [0.2, 0.4, 0.6, 0.8]
RATIO = ["i_leak_mean", "he_flow_mean", "he_p_set_mean", "he_p_mean", "i_leak_max"]
DIFF = ["ch_press_mean", "ch_press_max", "t_wafer_mean", "dechuck_time_s", "v_rb_min_frac", "t_zone_spread"]
SERIES = RATIO + DIFF + ["dz_mean", "dz_dev"]
STATS = ["now", "d30", "d60", "slope", "jump"]
FEATURE_NAMES = ([f"{s}_{k}" for s in SERIES for k in STATS] +
                 [f"dz{k}_now" for k in range(1, 5)] +
                 ["p_gap_now", "pset_per_dz", "press_per_flow", "leak_per_flow", "stick_rate", "spike_rate",
                  "frac_unseen_recipe", "wafers_since_install", "wafers_since_clean"])


def normalise(w):
    """Per-wafer series of one chuck relative to its own reference wafers (same recipe)."""
    ref = w.iloc[:REF_WAFERS]
    zk = ["t_z1_mean", "t_z2_mean", "t_z3_mean", "t_z4_mean"]
    cols = RATIO + DIFF + zk
    base = ref.groupby("recipe")[cols].median().reindex(w.recipe.values)
    base.index = w.index
    out = pd.DataFrame(index=w.index)
    for c in RATIO:
        out[c] = w[c] / base[c]
    for c in DIFF:
        out[c] = w[c] - base[c]
    dz = np.column_stack([w[c].values - base[c].values for c in zk])
    for k in range(4):
        out[f"dz{k + 1}"] = dz[:, k]
    out["dz_mean"] = dz.mean(axis=1)
    dev = dz - dz.mean(axis=1, keepdims=True)
    idx = np.nan_to_num(np.abs(dev), nan=-1).argmax(axis=1)
    out["dz_dev"] = dev[np.arange(len(dev)), idx]
    out["unseen"] = base[RATIO[0]].isna().astype(float).values
    # spike flag: peak leakage well above this chuck's reference peak for the recipe
    out["spike"] = (out["i_leak_max"] > 1.25).astype(float)
    out["stick"] = (w["dechuck_time_s"].values > (base["dechuck_time_s"].values + 2.0)).astype(float)
    return out


def _stats(v):
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return [np.nan] * 5
    now = np.median(v[-30:])
    p1 = np.median(v[-60:-30]) if n > 40 else now
    p2 = np.median(v[-90:-60]) if n > 70 else p1
    if n > 10:
        x = np.arange(n) - (n - 1) / 2
        slope = float((x * (v - v.mean())).sum() / (x * x).sum() * 30)
    else:
        slope = 0.0
    if n >= 20:
        med = np.array([np.median(v[i:i + 10]) for i in range(0, n - 9, 5)])
        jump = float(np.max(np.abs(np.diff(med)))) if len(med) > 1 else 0.0
    else:
        jump = 0.0
    return [float(now), float(now - p1), float(p1 - p2), slope, jump]


def window_features(w, norm, end):
    lo = max(REF_WAFERS, end + 1 - 90)
    seg = norm.iloc[lo:end + 1]
    cur = norm.iloc[max(REF_WAFERS, end + 1 - 30):end + 1]
    f = []
    for s in SERIES:
        f += _stats(seg[s].values)
    g = dict(zip([f"{s}_{k}" for s in SERIES for k in STATS], f))
    f += [float(np.nanmedian(cur[f"dz{k}"].values)) if cur[f"dz{k}"].notna().any() else np.nan for k in range(1, 5)]
    pset = g["he_p_set_mean_now"] - 1
    f += [g["he_p_mean_now"] - g["he_p_set_mean_now"], pset / (abs(g["dz_mean_now"]) + 0.15),
          g["ch_press_mean_now"] / (abs(g["he_flow_mean_now"] - 1) + 0.1),
          (g["i_leak_mean_now"] - 1) / (abs(g["he_flow_mean_now"] - 1) + 0.1),
          float(cur["stick"].mean()), float(cur["spike"].mean()), float(cur["unseen"].mean()),
          float(w.wafers_since_install.iloc[end]), float(w.wafers_since_clean.iloc[end])]
    return f


def feature_matrix(wafers, rows):
    """rows: DataFrame with chuck_id, end_wafer."""
    groups = {c: g.sort_values("wafer_idx").reset_index(drop=True) for c, g in wafers.groupby("chuck_id")}
    norms = {c: normalise(g) for c, g in groups.items()}
    return np.array([window_features(groups[r.chuck_id], norms[r.chuck_id], int(r.end_wafer)) for r in rows.itertuples()],
                    dtype=float)


def to_multihot(labels):
    """'he_leak+heater_zone' -> 0/1 vector over FAULTS; 'healthy' -> all zeros."""
    Y = np.zeros((len(labels), len(FAULTS)), dtype=int)
    for i, l in enumerate(labels):
        if l != "healthy":
            for f in l.split("+"):
                Y[i, FAULTS.index(f)] = 1
    return Y


def from_multihot(Y):
    names = np.array(FAULTS)
    return ["+".join(names[r.astype(bool)]) if r.any() else "healthy" for r in Y]
