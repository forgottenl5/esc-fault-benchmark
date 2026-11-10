"""Generate the ESC fault benchmark.

usage: python generate.py --out data [--test-seed-file private/test_seed.txt]

Train/validation chucks use the public seed below, so anyone can rebuild them.
Test chucks use a private seed that is not in the repository; without it the test
labels cannot be regenerated. Test labels are written to <out>/private_test_labels/.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from escbench.sim import simulate_chuck, FAULTS, CHANNELS, T
from escbench.features import summarize, SUMMARY_COLS

PUBLIC_SEED = 20261109
PER_CLASS = {"train": 14, "val": 2, "test": 4}


def chuck_plan():
    plan = []
    cid = 0
    for split in ["train", "val", "test"]:
        for f in FAULTS:
            for _ in range(PER_CLASS[split]):
                plan.append((cid, split, f)); cid += 1
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-seed-file", default="private/test_seed.txt")
    ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()
    splits = a.splits.split(",")
    test_seed = None
    if "test" in splits:
        if not os.path.exists(a.test_seed_file):
            raise SystemExit(f"test seed file {a.test_seed_file} missing; generate train,val only")
        test_seed = int(open(a.test_seed_file).read().strip())

    plan = [p for p in chuck_plan() if p[1] in splits]
    # shuffle ids within split so the chuck id does not reveal the class
    by_split = {}
    for cid, split, f in plan:
        by_split.setdefault(split, []).append(f)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "private_test_labels"), exist_ok=True)

    next_id = {"train": 1000, "val": 2000, "test": 3000}
    for split, flist in by_split.items():
        seed = test_seed if split == "test" else PUBLIC_SEED
        order_rng = np.random.default_rng([seed, ord(split[0])])
        flist = list(order_rng.permutation(flist))
        ts_frames, wafer_frames, label_frames, metas = [], [], [], []
        for j, f in enumerate(flist):
            chuck_id = next_id[split] + j
            rng = np.random.default_rng([seed, chuck_id])
            r = simulate_chuck(rng, f, chuck_id)
            N = r["meta"]["n_wafers"]
            ts = r["ts"]
            # long-format raw time series
            d = {"chuck_id": np.full(N * T, chuck_id, np.int32),
                 "wafer_idx": np.repeat(np.arange(N, dtype=np.int32), T),
                 "t_s": np.tile(np.arange(T, dtype=np.int16), N)}
            for k, c in enumerate(CHANNELS):
                d[c] = ts[:, :, k].reshape(-1)
            ts_frames.append(pd.DataFrame(d))
            w = r["wafer"]
            S = summarize(ts)
            wd = {"chuck_id": np.full(N, chuck_id, np.int32), "wafer_idx": w["wafer_idx"],
                  "recipe": w["recipe"], "wafers_since_install": w["wafers_since_install"],
                  "wafers_since_clean": w["wafers_since_clean"], "dechuck_time_s": w["dechuck_time_s"]}
            for k, c in enumerate(SUMMARY_COLS):
                wd[c] = S[:, k]
            wafer_frames.append(pd.DataFrame(wd))
            label_frames.append(pd.DataFrame({
                "chuck_id": np.full(N, chuck_id, np.int32), "wafer_idx": w["wafer_idx"],
                "fault": f, "severity": w["severity"], "rul": w["rul"]}))
            metas.append(r["meta"])
            print(split, chuck_id, f, N, flush=True)
        pd.concat(ts_frames).to_parquet(os.path.join(a.out, f"timeseries_{split}.parquet"),
                                        index=False, row_group_size=200_000)
        pd.concat(wafer_frames).to_parquet(os.path.join(a.out, f"wafers_{split}.parquet"), index=False)
        lab_dir = os.path.join(a.out, "private_test_labels") if split == "test" else a.out
        pd.concat(label_frames).to_parquet(os.path.join(lab_dir, f"labels_{split}.parquet"), index=False)
        meta_clean = [{k: (str(v) if isinstance(v, np.str_) else v) for k, v in m.items()} for m in metas]
        with open(os.path.join(lab_dir, f"chucks_{split}.json"), "w") as fh:
            json.dump(meta_clean, fh, indent=1)


if __name__ == "__main__":
    main()
