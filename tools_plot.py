"""Sanity plots: one wafer's raw signals, and per-wafer trends for one chuck of each fault class."""
import json, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = sys.argv[1] if len(sys.argv) > 1 else "data"
w = pd.read_parquet(f"{d}/wafers_train.parquet"); lab = pd.read_parquet(f"{d}/labels_train.parquet")
meta = json.load(open(f"{d}/chucks_train.json"))
ts = pd.read_parquet(f"{d}/timeseries_train.parquet", filters=[("chuck_id", "==", meta[0]["chuck_id"]), ("wafer_idx", "==", 5)])
fig, ax = plt.subplots(2, 3, figsize=(13, 6))
for a, cols in zip(ax.flat, [["v_set", "v_rb"], ["i_leak_ma"], ["he_p_set_torr", "he_p_torr", "he_flow_sccm"],
                             ["t_wafer_c", "t_z1_c", "t_z2_c", "t_z3_c", "t_z4_c"], ["rf_power_w"], ["ch_press_mtorr"]]):
    for c in cols: a.plot(ts.t_s, ts[c], label=c)
    a.legend(fontsize=7); a.set_xlabel("time in wafer (s)")
fig.suptitle(f"One wafer, healthy chuck {meta[0]['chuck_id']}"); fig.tight_layout(); fig.savefig("figures/one_wafer.png", dpi=110)
feats = ["i_leak_mean", "he_p_set_mean", "he_flow_mean", "ch_press_mean", "t_zone_spread", "dechuck_time_s", "he_p_mean"]
faults = ["healthy", "contamination", "wear", "he_leak", "residual_charge", "heater_zone", "sensor_drift"]
fig, ax = plt.subplots(len(faults), len(feats), figsize=(18, 15), sharex="row")
for i, f in enumerate(faults):
    m = next(m for m in meta if m["fault"] == f)
    x = w[w.chuck_id == m["chuck_id"]]; x = x[x.recipe == 0]
    for j, c in enumerate(feats):
        ax[i, j].plot(x.wafer_idx, x[c], ".", ms=2)
        if m["onset_wafer"] >= 0: ax[i, j].axvline(m["onset_wafer"], color="r", lw=0.8)
        if i == 0: ax[i, j].set_title(c, fontsize=9)
    ax[i, 0].set_ylabel(f + (f"\n{m['drift_sensor']}" if m["drift_sensor"] else ""), fontsize=9)
fig.suptitle("Per-wafer trends (recipe 0 only); red line = fault onset"); fig.tight_layout(); fig.savefig("figures/fault_trends.png", dpi=80)
