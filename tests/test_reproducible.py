"""Rebuild the validation split from the public seed and check it matches the published files exactly."""
import os, subprocess, sys, tempfile
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLISHED = os.path.join(ROOT, "data_small")


def test_val_split_rebuilds_exactly():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(ROOT, "generate.py"), "--out", tmp, "--splits", "val"],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        for name in ["wafers_val.parquet", "labels_val.parquet"]:
            a = pd.read_parquet(os.path.join(tmp, name))
            b = pd.read_parquet(os.path.join(PUBLISHED, name))
            pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    test_val_split_rebuilds_exactly()
    print("OK: validation split rebuilds exactly from the public seed")
