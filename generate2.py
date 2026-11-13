"""Generate ESC-Fault v2.

usage:
  python generate2.py --out data_v2 --splits train,val                    # rebuild public splits exactly (tables only)
  python generate2.py --out data_v2 --splits train --raw                  # also write raw 1 Hz time series
  python generate2.py --out dev --splits train --per-class 100 --seed 1   # make your own, larger training set

Splits (tool A unless noted): train, val, test, test_shift (tool B: other recipes, hardware and sensor noise).
Public splits use PUBLIC_SEED; test and test_shift use a private seed (not in the repository), so their
answers cannot be rebuilt from the code. Their labels go to <out>/private_test_labels/.
"""
import argparse, json, os
import numpy as np
import pandas as pd
from escbench.sim2 import simulate_chuck, FAULTS, CHANNELS, T
from escbench.features_v2 import summarize, quantize, SUMMARY_COLS

PUBLIC_SEED = 20261112
SEV_LABEL = 0.15
PAIRS = [("contamination", "sensor_drift"), ("he_leak", "heater_zone"), ("wear", "arcing"),
         ("residual_charge", "sensor_drift"), ("he_leak", "sensor_drift"), ("heater_zone", "arcing"),
         ("contamination", "he_leak"), ("wear", "residual_charge")]
# chucks per single-fault class (incl. healthy) and per overlapping pair
SIZES = {"train": (40, 5), "val": (5, 1), "test": (10, 2), "test_shift": (5, 1)}
ID_BASE = {"train": 10000, "val": 20000, "test": 30000, "test_shift": 40000}


def plan_split(split, per_class=None, per_pair=None):
    pc, pp = SIZES[split]
    pc = per_class if per_class is not None else pc
    pp = per_pair if per_pair is not None else pp
    groups = [[]] * pc + [[f] for f in FAULTS for _ in range(pc)] + [list(p) for p in PAIRS for _ in range(pp)]
    return groups


def label_of(sev_row):
    act = sorted(f for f in FAULTS if sev_row[f] >= SEV_LABEL)
    return "+".join(act) if act else "healthy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_v2")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--seed", type=int, default=None, help="override the seed (for your own extra data)")
    ap.add_argument("--test-seed-file", default="private/test_seed_v2.txt")
    ap.add_argument("--per-class", type=int, default=None)
    ap.add_argument("--per-pair", type=int, default=None)
    ap.add_argument("--raw", action="store_true", help="also write raw 1 Hz time series")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split in a.splits.split(","):
        private = split in ("test", "test_shift")
        if a.seed is not None:
            seed = a.seed
        elif private:
            if not os.path.exists(a.test_seed_file):
                raise SystemExit(f"{a.test_seed_file} missing; test splits need the private seed")
            seed = int(open(a.test_seed_file).read().strip())
        else:
            seed = PUBLIC_SEED
        tool = "B" if split == "test_shift" else "A"
        groups = plan_split(split, a.per_class, a.per_pair)
        order = np.random.default_rng([seed, ID_BASE[split]]).permutation(len(groups))
        W, L, R, M = [], [], [], []
        for j, gi in enumerate(order):
            cid = ID_BASE[split] + j
            rng = np.random.default_rng([seed, cid])
            new_at = int(rng.integers(300, 650)) if rng.random() < 0.3 else None
            r = simulate_chuck(rng, groups[gi], cid, tool=tool, new_recipe_at=new_at)
            ts = quantize(r["ts"]); w = r["wafer"]; N = r["meta"]["n_wafers"]
            S = summarize(ts)
            d = {"chuck_id": np.full(N, cid, np.int32), "wafer_idx": w["wafer_idx"], "recipe": w["recipe"],
                 "wafers_since_install": w["wafers_since_install"], "wafers_since_clean": w["wafers_since_clean"],
                 "dechuck_time_s": w["dechuck_time_s"]}
            for k, c in enumerate(SUMMARY_COLS):
                d[c] = S[:, k]
            W.append(pd.DataFrame(d))
            sev = pd.DataFrame({f: w[f"sev_{f}"] for f in FAULTS})
            lab = pd.DataFrame({"chuck_id": np.full(N, cid, np.int32), "wafer_idx": w["wafer_idx"]})
            for f in FAULTS:
                lab[f"sev_{f}"] = sev[f].values
            act = sev.values >= SEV_LABEL
            names = np.array(FAULTS)
            lab["label"] = ["+".join(names[row]) if row.any() else "healthy" for row in act]
            lab["rul"] = w["rul"]
            L.append(lab)
            if a.raw or private:
                dd = {"chuck_id": np.full(N * T, cid, np.int32), "wafer_idx": np.repeat(np.arange(N, dtype=np.int32), T),
                      "t_s": np.tile(np.arange(T, dtype=np.int16), N)}
                for k, c in enumerate(CHANNELS):
                    dd[c] = ts[:, :, k].reshape(-1)
                R.append(pd.DataFrame(dd))
            M.append(r["meta"])
        lab_dir = os.path.join(a.out, "private_test_labels") if private else a.out
        os.makedirs(lab_dir, exist_ok=True)
        pd.concat(W).to_parquet(os.path.join(a.out, f"wafers_{split}.parquet"), index=False, compression="zstd")
        pd.concat(L).to_parquet(os.path.join(lab_dir, f"labels_{split}.parquet"), index=False, compression="zstd")
        json.dump(M, open(os.path.join(lab_dir, f"chucks_{split}.json"), "w"), indent=1)
        if R:
            pd.concat(R).to_parquet(os.path.join(a.out, f"timeseries_{split}.parquet"), index=False,
                                    compression="zstd", row_group_size=500_000)
        print(split, "chucks", len(M), "wafers", sum(m["n_wafers"] for m in M), flush=True)


if __name__ == "__main__":
    main()
