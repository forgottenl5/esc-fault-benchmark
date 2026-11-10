"""Build task files from generated data. Test answers go to <data>/private_test_labels/.

usage: python make_tasks.py --data data --out tasks
"""
import argparse, json, os
import numpy as np
import pandas as pd
from escbench.tasks import SEV_LABEL, REF_WAFERS, RUL_FRACS

WINDOWS_PER_CHUCK = {"train": 60, "val": 20, "test": 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="tasks")
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    priv = os.path.join(a.data, "private_test_labels")
    for split in a.splits.split(","):
        ldir = priv if split == "test" else a.data
        lab = pd.read_parquet(os.path.join(ldir, f"labels_{split}.parquet"))
        meta = {m["chuck_id"]: m for m in json.load(open(os.path.join(ldir, f"chucks_{split}.json")))}
        rng = np.random.default_rng(7 + ord(split[0]))
        t1, t2, t3 = [], [], []
        for cid, g in lab.groupby("chuck_id"):
            g = g.sort_values("wafer_idx"); m = meta[cid]; N = len(g); sev = g.severity.values
            # task 1: windows; ambiguous windows (0 < severity < SEV_LABEL) are skipped
            ends = rng.choice(np.arange(REF_WAFERS + 29, N), size=min(WINDOWS_PER_CHUCK[split] * 3, N - REF_WAFERS - 29), replace=False)
            kept = 0
            for e in ends:
                s = sev[e]
                if 0 < s < SEV_LABEL:
                    continue
                label = m["fault"] if (m["fault"] != "healthy" and s >= SEV_LABEL) else "healthy"
                t1.append((cid, int(e), label)); kept += 1
                if kept >= WINDOWS_PER_CHUCK[split]:
                    break
            # task 2: one row per chuck
            t2.append((cid, N, m["fault"], m["onset_wafer"]))
            # task 3: remaining life at cut points, run-to-failure chucks only
            if m["run_to_failure"]:
                on = m["onset_wafer"]
                for fr in RUL_FRACS:
                    cut = int(on + fr * (N - 1 - on))
                    t3.append((cid, cut, int(N - 1 - cut)))
            elif split == "test":
                # decoy cut points on other chucks, so the task-3 file does not reveal which test
                # chucks wear out (that would leak task-1 answers); decoys are not scored
                for cut in sorted(rng.choice(np.arange(200, N - 1), size=len(RUL_FRACS), replace=False)):
                    t3.append((cid, int(cut), -1))
        t1 = pd.DataFrame(t1, columns=["chuck_id", "end_wafer", "label"])
        t2 = pd.DataFrame(t2, columns=["chuck_id", "n_wafers", "fault", "onset_wafer"])
        t3 = pd.DataFrame(t3, columns=["chuck_id", "end_wafer", "rul"])
        if split == "test":
            t1.to_csv(os.path.join(priv, "task1_answers.csv"), index=False)
            t2.to_csv(os.path.join(priv, "task2_answers.csv"), index=False)
            t3[t3.rul >= 0].to_csv(os.path.join(priv, "task3_answers.csv"), index=False)
            t1 = t1.drop(columns=["label"]); t2 = t2[["chuck_id", "n_wafers"]]; t3 = t3.drop(columns=["rul"])
        t1.to_csv(os.path.join(a.out, f"task1_{split}.csv"), index=False)
        t2.to_csv(os.path.join(a.out, f"task2_{split}.csv"), index=False)
        t3.to_csv(os.path.join(a.out, f"task3_{split}.csv"), index=False)
        print(split, "task1", len(t1), "task2", len(t2), "task3", len(t3))


if __name__ == "__main__":
    main()
