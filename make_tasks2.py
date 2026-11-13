"""Build v2 task files. Answers for test and test_shift go to <data>/private_test_labels/.

usage: python make_tasks2.py --data data_v2 --out tasks_v2 --splits train,val
"""
import argparse, json, os
import numpy as np
import pandas as pd
from escbench.sim2 import FAULTS
from escbench.tasks_v2 import SEV_LABEL, REF_WAFERS, RUL_FRACS

WINDOWS = {"train": 40, "val": 20, "test": 20, "test_shift": 20}
SEV_COLS = [f"sev_{f}" for f in FAULTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v2")
    ap.add_argument("--out", default="tasks_v2")
    ap.add_argument("--splits", default="train,val,test,test_shift")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    priv = os.path.join(a.data, "private_test_labels")
    for split in a.splits.split(","):
        private = split in ("test", "test_shift")
        ldir = priv if private else a.data
        lab = pd.read_parquet(os.path.join(ldir, f"labels_{split}.parquet"))
        meta = {m["chuck_id"]: m for m in json.load(open(os.path.join(ldir, f"chucks_{split}.json")))}
        rng = np.random.default_rng([11, len(split), ord(split[-1])])
        t1, t2, t3 = [], [], []
        for cid, g in lab.groupby("chuck_id"):
            g = g.sort_values("wafer_idx"); m = meta[cid]; N = len(g)
            sev = g[SEV_COLS].values; labels = g.label.values
            ends = rng.choice(np.arange(REF_WAFERS + 29, N), size=min(WINDOWS[split] * 3, N - REF_WAFERS - 29), replace=False)
            kept = 0
            for e in ends:
                if ((sev[e] > 0) & (sev[e] < SEV_LABEL)).any():
                    continue          # a fault is starting but still below the labelling threshold: ambiguous, skipped
                t1.append((cid, int(e), labels[e])); kept += 1
                if kept >= WINDOWS[split]:
                    break
            on = m["onsets"]
            first = min(on.values()) if on else -1
            t2.append((cid, N, "+".join(sorted(m["faults"])) if m["faults"] else "healthy", first,
                       json.dumps(on)))
            if m["run_to_failure"]:
                f = [x for x in m["faults"] if x in ("contamination", "wear")][0]
                o = on[f]
                for fr in RUL_FRACS:
                    cut = int(o + fr * (N - 1 - o))
                    t3.append((cid, cut, int(N - 1 - cut)))
            elif private:
                for cut in sorted(rng.choice(np.arange(200, N - 1), size=len(RUL_FRACS), replace=False)):
                    t3.append((cid, int(cut), -1))   # decoys, not scored: hide which chucks wear out
        t1 = pd.DataFrame(t1, columns=["chuck_id", "end_wafer", "label"])
        t2 = pd.DataFrame(t2, columns=["chuck_id", "n_wafers", "faults", "first_onset", "onsets"])
        t3 = pd.DataFrame(t3, columns=["chuck_id", "end_wafer", "rul"])
        if private:
            t1.to_csv(os.path.join(priv, f"task1_{split}_answers.csv"), index=False)
            t2.to_csv(os.path.join(priv, f"task2_{split}_answers.csv"), index=False)
            t3[t3.rul >= 0].to_csv(os.path.join(priv, f"task3_{split}_answers.csv"), index=False)
            t1 = t1.drop(columns=["label"]); t2 = t2[["chuck_id", "n_wafers"]]; t3 = t3.drop(columns=["rul"])
        t1.to_csv(os.path.join(a.out, f"task1_{split}.csv"), index=False)
        t2.to_csv(os.path.join(a.out, f"task2_{split}.csv"), index=False)
        t3.to_csv(os.path.join(a.out, f"task3_{split}.csv"), index=False)
        print(split, "task1", len(t1), "task2", len(t2), "task3", len(t3))


if __name__ == "__main__":
    main()
