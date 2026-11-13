"""Score v2 predictions.

usage: python score2.py --task 1 --pred my_task1.csv --answers tasks_v2/task1_val.csv
       python score2.py --task 2 --pred my_task2.csv --answers tasks_v2/task2_val.csv
       python score2.py --task 3 --pred my_task3.csv --answers tasks_v2/task3_val.csv

Prediction files:
  task 1: chuck_id,end_wafer,label        label = "healthy" or active faults joined by "+", e.g. "he_leak+heater_zone"
  task 2: chuck_id,alarm_wafer,fault      one row per chuck; alarm_wafer = -1 for no alarm; fault = one fault name
  task 3: chuck_id,end_wafer,rul_pred
"""
import argparse, json
import numpy as np
import pandas as pd
from escbench.sim2 import FAULTS
from escbench.tasks_v2 import to_multihot


def score_task1(pred, ans):
    m = ans.merge(pred, on=["chuck_id", "end_wafer"], how="left", suffixes=("", "_pred"))
    m["label_pred"] = m["label_pred"].fillna("healthy")
    Y, P = to_multihot(m.label.values), to_multihot(m.label_pred.values)
    f1 = {}
    for k, f in enumerate(FAULTS):
        tp = int((Y[:, k] & P[:, k]).sum()); fp = int((~Y[:, k].astype(bool) & P[:, k].astype(bool)).sum())
        fn = int((Y[:, k].astype(bool) & ~P[:, k].astype(bool)).sum())
        f1[f] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    healthy = Y.sum(1) == 0
    return {"macro_f1": float(np.mean(list(f1.values()))), "exact_match": float((Y == P).all(1).mean()),
            "healthy_false_alarm_rate": float((P[healthy].sum(1) > 0).mean()) if healthy.any() else None,
            "per_fault_f1": {k: round(v, 3) for k, v in f1.items()}}


def score_task2(pred, ans):
    """A chuck's alarm counts as a detection if it comes at or after the onset of one of its faults and names
    that fault; an alarm before every onset, or any alarm on a healthy chuck, is a false alarm."""
    m = ans.merge(pred.rename(columns={"fault": "pred_fault"}), on="chuck_id", how="left")
    m["alarm_wafer"] = m["alarm_wafer"].fillna(-1).astype(int)
    det, delays, false_alarms, missed, wrong, healthy_wafers, faulty = 0, [], 0, 0, 0, 0, 0
    for r in m.itertuples():
        on = json.loads(r.onsets)
        if not on:
            healthy_wafers += r.n_wafers
            false_alarms += int(r.alarm_wafer >= 0)
            continue
        faulty += 1
        first = min(on.values())
        healthy_wafers += first
        if r.alarm_wafer < 0:
            missed += 1
        elif r.alarm_wafer < first:
            false_alarms += 1
        elif r.pred_fault in on and r.alarm_wafer >= on[r.pred_fault]:
            det += 1; delays.append(r.alarm_wafer - on[r.pred_fault])
        else:
            wrong += 1
    return {"detection_rate": det / max(faulty, 1), "median_delay": float(np.median(delays)) if delays else None,
            "false_alarms_per_1k_healthy_wafers": 1000 * false_alarms / max(healthy_wafers, 1),
            "missed": missed, "wrong_fault": wrong}


def score_task3(pred, ans):
    m = ans.merge(pred, on=["chuck_id", "end_wafer"], how="left")
    m["rul_pred"] = m["rul_pred"].fillna(m["rul"].mean() if len(m) else 0)
    d = m.rul_pred - m.rul
    asym = np.where(d < 0, np.exp(-d / 130) - 1, np.exp(d / 100) - 1)
    return {"mae": float(np.abs(d).mean()), "asymmetric_score": float(asym.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--answers", required=True)
    a = ap.parse_args()
    fn = {1: score_task1, 2: score_task2, 3: score_task3}[a.task]
    print(json.dumps(fn(pd.read_csv(a.pred), pd.read_csv(a.answers)), indent=1))


if __name__ == "__main__":
    main()
