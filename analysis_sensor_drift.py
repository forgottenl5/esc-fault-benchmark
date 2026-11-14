"""Break down sensor-drift detection by which sensor drifts and by drift size (v2 baseline predictions).

usage: python analysis_sensor_drift.py --split val            (public labels)
       python analysis_sensor_drift.py --split test --labels-dir data_v2/private_test_labels   (maintainers only)
"""
import argparse, json, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val"); ap.add_argument("--data", default="data_v2")
    ap.add_argument("--tasks", default="tasks_v2"); ap.add_argument("--pred-dir", default="baselines_out_v2")
    ap.add_argument("--labels-dir", default=None)
    a = ap.parse_args()
    ldir = a.labels_dir or a.data
    meta = {m["chuck_id"]: m for m in json.load(open(os.path.join(ldir, f"chucks_{a.split}.json")))}
    ans_path = os.path.join(ldir, f"task1_{a.split}_answers.csv") if a.labels_dir else os.path.join(a.tasks, f"task1_{a.split}.csv")
    ans = pd.read_csv(ans_path)
    pred = pd.read_csv(os.path.join(a.pred_dir, f"{a.split}_task1_gbt.csv"))
    m = ans.merge(pred, on=["chuck_id", "end_wafer"], suffixes=("", "_pred"))
    m["true_sd"] = m.label.str.contains("sensor_drift")
    m["pred_sd"] = m.label_pred.str.contains("sensor_drift")
    m["sensor"] = m.chuck_id.map(lambda c: meta[c]["drift_sensor"] or "-")
    m["sensor"] = m.sensor.str.replace(r"t_z\d_c", "zone_temperature", regex=True)
    m["size"] = m.chuck_id.map(lambda c: meta[c]["magnitudes"].get("sensor_drift", np.nan))
    m["other_fault"] = m.chuck_id.map(lambda c: len(meta[c]["faults"]) > 1)
    sd = m[m.true_sd]
    print(f"== {a.split}: {sd.chuck_id.nunique()} sensor-drift chucks, {len(sd)} windows with active drift")
    print("recall by drifting sensor:")
    print(sd.groupby("sensor").agg(chucks=("chuck_id", "nunique"), windows=("pred_sd", "size"), recall=("pred_sd", "mean")).round(2))
    print("recall by drift size (fraction of full-scale drift):")
    print(sd.groupby(pd.cut(sd["size"], [0, 0.4, 0.7, 1.0])).agg(chucks=("chuck_id", "nunique"), recall=("pred_sd", "mean")).round(2))
    print("recall alone vs. with a second fault:")
    print(sd.groupby("other_fault").agg(chucks=("chuck_id", "nunique"), recall=("pred_sd", "mean")).round(2))
    miss = sd[~sd.pred_sd]
    print("what the model says instead when it misses:")
    print(miss.label_pred.value_counts().head(6))
    fp = m[~m.true_sd & m.pred_sd]
    print(f"false sensor_drift calls: {len(fp)} windows; on true labels:", fp.label.value_counts().head(4).to_dict())


if __name__ == "__main__":
    main()
