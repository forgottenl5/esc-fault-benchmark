"""Rebuild the v2 validation split from the public seed and check it matches the published checksums.

Checksums are content hashes of the tables (independent of the parquet writer version)."""
import json, os, subprocess, sys, tempfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def table_hash(df):
    return str(int(pd.util.hash_pandas_object(df, index=False).sum() % (2 ** 61)))


def test_v2_val_rebuilds_exactly(split="val"):
    expected = json.load(open(os.path.join(ROOT, "tests", "v2_checksums.json")))
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(ROOT, "generate2.py"), "--out", tmp, "--splits", split],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        for name in [f"wafers_{split}", f"labels_{split}"]:
            got = table_hash(pd.read_parquet(os.path.join(tmp, f"{name}.parquet")))
            assert got == expected[name], f"{name}: {got} != {expected[name]}"


if __name__ == "__main__":
    test_v2_val_rebuilds_exactly()
    print("OK: v2 validation split rebuilds exactly from the public seed")
