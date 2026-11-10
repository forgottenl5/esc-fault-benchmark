"""Score predictions for the three tasks.

usage:
  python score.py --task 1 --pred preds_task1.csv --answers tasks/task1_val.csv
  python score.py --task 2 --pred preds_task2.csv --answers tasks/task2_val.csv
  python score.py --task 3 --pred preds_task3.csv --answers tasks/task3_val.csv

Prediction files:
  task 1: chuck_id,end_wafer,label        (label = one of the 7 classes)
  task 2: chuck_id,alarm_wafer,fault      (alarm_wafer = -1 for no alarm; one alarm per chuck)
  task 3: chuck_id,end_wafer,rul_pred     (predicted wafers remaining after end_wafer)
"""
import argparse, json
import numpy as np
import pandas as pd
from escbench.tasks import FAULTS


def score_task1(pred, ans):
    m = ans.merge(pred, on=["chuck_id", "end_wafer"], how="left", suffixes=("", "_pred"))
    m["label_pred"] = m["label_pred"].fillna("healthy")
    f1s = {}
    for c in FAULTS:
        tp = ((m.label == c) & (m.label_pred == c)).sum()
        fp = ((m.label != c) & (m.label_pred == c)).sum()
        fn = ((m.label == c) & (m.label_pred != c)).sum()
        f1s[c] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return {"macro_f1": float(np.mean(list(f1s.values()))), "accuracy": float((m.label == m.label_pred).mean()),
            "per_class_f1": {k: round(v, 3) for k, v in f1s.items()}}


def score_task2(pred, ans):
    """One alarm per chuck. An alarm before onset (or on a healthy chuck) is a false alarm.
    An alarm at/after onset with the right fault type is a detection; with the wrong type, a misdiagnosis."""
    pred = pred.rename(columns={"fault": "pred_fault"})
    m = ans.rename(columns={"fault": "true_fault"}).merge(pred, on="chuck_id", how="left")
    m["alarm_wafer"] = m["alarm_wafer"].fillna(-1).astype(int)
    healthy_wafers = false_alarms = detected = wrong_type = faulty = 0
    delays = []
    for r in m.itertuples():
        faulty_chuck = r.true_fault != "healthy"
        on = r.onset_wafer if faulty_chuck else r.n_wafers
        healthy_wafers += on
        faulty += faulty_chuck
        if r.alarm_wafer < 0:
            continue
        if r.alarm_wafer < on:
            false_alarms += 1
        elif r.pred_fault == r.true_fault:
            detected += 1
            delays.append(r.alarm_wafer - on)
        else:
            wrong_type += 1
    return {"detection_rate": detected / faulty,
            "median_delay_wafers": float(np.median(delays)) if delays else None,
            "misdiagnosed": wrong_type, "false_alarms": false_alarms,
            "false_alarms_per_1000_healthy_wafers": round(1000 * false_alarms / healthy_wafers, 3)}


def score_task3(pred, ans):
    m = ans.merge(pred, on=["chuck_id", "end_wafer"], how="left")
    m["rul_pred"] = m["rul_pred"].fillna(0)
    d = m.rul_pred - m.rul                              # positive = predicted too late (overestimate)
    pen = np.where(d < 0, np.exp(-d / 130.0) - 1, np.exp(d / 100.0) - 1)
    return {"mae_wafers": float(np.abs(d).mean()), "asym_score": float(pen.mean()), "n": int(len(m))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--answers", required=True)
    a = ap.parse_args()
    pred, ans = pd.read_csv(a.pred), pd.read_csv(a.answers)
    out = {1: score_task1, 2: score_task2, 3: score_task3}[a.task](pred, ans)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
