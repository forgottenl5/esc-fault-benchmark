"""Extended causal features (v2 baseline): trends, step size, and cross-signal consistency.

Everything uses only wafers 0..end of one chuck, like escbench.tasks.window_features.
"""
import numpy as np
import pandas as pd
from .tasks import REF_WAFERS, WINDOW, RATIO, DIFF, FEATURE_NAMES, window_features

SERIES_RATIO = ["i_leak_mean", "he_flow_mean", "he_p_set_mean"]
SERIES_DIFF = ["ch_press_mean", "t_wafer_mean", "dz_mean", "dz_max", "dechuck_time_s"]
LONG = 90


def _normalised(w):
    """Per-wafer series relative to the chuck's own first-100-wafer reference, recipe by recipe."""
    ref = w.iloc[:REF_WAFERS]
    cols = list(set(RATIO + DIFF + ["dechuck_time_s"]))
    med = ref.groupby("recipe")[cols].median()
    out = pd.DataFrame(index=w.index)
    rec = w.recipe.values
    base = med.reindex(rec)
    base.index = w.index
    for c in SERIES_RATIO:
        out[c] = w[c] / base[c]
    zk = ["t_z1_mean", "t_z2_mean", "t_z3_mean", "t_z4_mean"]
    dz = np.column_stack([w[c].values - base[c].values for c in zk])
    out["dz_mean"] = dz.mean(axis=1)
    dev = dz - dz.mean(axis=1, keepdims=True)
    out["dz_max"] = dev[np.arange(len(dev)), np.abs(dev).argmax(axis=1)]
    for c in ["ch_press_mean", "t_wafer_mean", "dechuck_time_s"]:
        out[c] = w[c] - base[c]
    return out


NAMES2 = ([f"{s}_{k}" for s in SERIES_RATIO + SERIES_DIFF for k in ("now", "d30", "d60", "slope", "jump")] +
          ["dz_mean_now_x_pset", "pset_per_dz", "press_per_flow", "leak_per_flow", "wafers_since_clean"])


def window_features2(w, end, norm=None):
    base = window_features(w, end)
    if norm is None:
        norm = _normalised(w)
    lo = max(REF_WAFERS, end + 1 - LONG)
    seg = norm.iloc[lo:end + 1]
    f = []
    for c in SERIES_RATIO + SERIES_DIFF:
        v = seg[c].values
        n = len(v)
        now = np.median(v[-30:]) if n else 0.0
        p1 = np.median(v[-60:-30]) if n > 40 else now
        p2 = np.median(v[-90:-60]) if n > 70 else p1
        x = np.arange(n) - (n - 1) / 2
        slope = float((x * (v - v.mean())).sum() / max((x * x).sum(), 1e-9) * 30) if n > 10 else 0.0
        k = 10
        med = np.array([np.median(v[i:i + k]) for i in range(0, max(n - k + 1, 1), 5)]) if n >= 2 * k else np.array([now, now])
        jump = float(np.max(np.abs(np.diff(med)))) if len(med) > 1 else 0.0
        f += [float(now), float(now - p1), float(p1 - p2), slope, jump]
    g = dict(zip([f"{s}_{k}" for s in SERIES_RATIO + SERIES_DIFF for k in ("now", "d30", "d60", "slope", "jump")], f))
    dz, ps = g["dz_mean_now"], g["he_p_set_mean_now"] - 1
    f += [dz * ps, ps / (abs(dz) + 0.15),
          g["ch_press_mean_now"] / (abs(g["he_flow_mean_now"] - 1) + 0.1),
          (g["i_leak_mean_now"] - 1) / (abs(g["he_flow_mean_now"] - 1) + 0.1),
          float(w.wafers_since_clean.iloc[end])]
    return list(base) + f


FEATURE_NAMES2 = FEATURE_NAMES + NAMES2


def feature_matrix2(wafers, rows):
    groups = {cid: g.sort_values("wafer_idx").reset_index(drop=True) for cid, g in wafers.groupby("chuck_id")}
    norms = {cid: _normalised(g) for cid, g in groups.items()}
    return np.array([window_features2(groups[r.chuck_id], int(r.end_wafer), norms[r.chuck_id]) for r in rows.itertuples()])
