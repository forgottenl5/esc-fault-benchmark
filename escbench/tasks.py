"""Task definitions and a reference feature extractor shared by make_tasks.py, baselines.py and score.py."""
import numpy as np
import pandas as pd

FAULTS = ["healthy", "contamination", "wear", "he_leak", "residual_charge", "heater_zone", "sensor_drift"]
REF_WAFERS = 100        # first wafers of every chuck are healthy (all onsets are at wafer 150 or later)
WINDOW = 30             # wafers summarised for the "current" state
SEV_LABEL = 0.15        # task 1: a window counts as faulty once severity >= 0.15
RUL_FRACS = [0.2, 0.4, 0.6, 0.8]

RATIO = ["i_leak_mean", "he_p_set_mean", "he_p_mean", "he_flow_mean"]
DIFF = ["ch_press_mean", "t_zone_spread", "t_wafer_mean", "t_z1_mean", "t_z2_mean", "t_z3_mean", "t_z4_mean"]
FEATURE_NAMES = ([f"r_{c}" for c in RATIO] + [f"d_{c}" for c in DIFF] +
                 ["d_dechuck_med", "stick_rate", "d_p_gap", "zone_dev_max", "flow_per_press", "wafers_since_install"])


def _per_recipe(df, cols, agg):
    return df.groupby("recipe")[cols].agg(agg)


def window_features(w, end):
    """w: per-wafer table of ONE chuck (sorted by wafer_idx). Uses wafers <= end only."""
    ref = w.iloc[:REF_WAFERS]
    cur = w.iloc[max(REF_WAFERS, end + 1 - WINDOW):end + 1]
    if len(cur) == 0:
        cur = w.iloc[end + 1 - WINDOW:end + 1]
    cols = RATIO + DIFF + ["dechuck_time_s"]
    rr, cc = _per_recipe(ref, cols, "median"), _per_recipe(cur, cols, "median")
    common = cc.index.intersection(rr.index)
    rr, cc = rr.loc[common], cc.loc[common]
    weight = cur.recipe.value_counts().reindex(common).values.astype(float)
    weight /= weight.sum()
    f = {}
    for c in RATIO:
        f[f"r_{c}"] = float(np.dot(weight, cc[c] / rr[c]))
    for c in DIFF:
        f[f"d_{c}"] = float(np.dot(weight, cc[c] - rr[c]))
    f["d_dechuck_med"] = float(np.dot(weight, cc["dechuck_time_s"] - rr["dechuck_time_s"]))
    base = ref.dechuck_time_s.median() + 2.0
    f["stick_rate"] = float((cur.dechuck_time_s > base).mean())
    f["d_p_gap"] = f["r_he_p_mean"] - f["r_he_p_set_mean"]
    f["zone_dev_max"] = max(abs(f[f"d_t_z{k}_mean"] - np.mean([f[f"d_t_z{j}_mean"] for j in range(1, 5)]))
                            for k in range(1, 5))
    f["flow_per_press"] = f["r_he_flow_mean"] / max(f["r_he_p_mean"], 1e-6)
    f["wafers_since_install"] = float(w.wafers_since_install.iloc[end])
    return [f[k] for k in FEATURE_NAMES]


def feature_matrix(wafers, rows):
    """rows: DataFrame with chuck_id, end_wafer. Returns np.array (len(rows), n_features)."""
    groups = {cid: g.sort_values("wafer_idx").reset_index(drop=True) for cid, g in wafers.groupby("chuck_id")}
    return np.array([window_features(groups[r.chuck_id], int(r.end_wafer)) for r in rows.itertuples()])
