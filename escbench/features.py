"""Per-wafer summary features computed from the raw time series (what a fab's FDC system would log)."""
import numpy as np
from .sim import CHANNELS

IDX = {c: i for i, c in enumerate(CHANNELS)}
STEADY = slice(15, 78)   # RF-on steady window (s)

SUMMARY_COLS = ["v_rb_mean", "i_leak_mean", "he_p_set_mean", "he_p_mean", "he_flow_mean",
                "t_wafer_mean", "t_wafer_max", "t_z1_mean", "t_z2_mean", "t_z3_mean", "t_z4_mean",
                "t_zone_spread", "t_cool_mean", "rf_power_mean", "ch_press_mean"]


def summarize(ts):
    """ts: (N, T, C) -> (N, len(SUMMARY_COLS)) float32"""
    st = ts[:, STEADY, :].mean(axis=1)
    z = st[:, IDX["t_z1_c"]:IDX["t_z4_c"] + 1]
    out = np.column_stack([
        st[:, IDX["v_rb"]], st[:, IDX["i_leak_ma"]], st[:, IDX["he_p_set_torr"]], st[:, IDX["he_p_torr"]],
        st[:, IDX["he_flow_sccm"]], st[:, IDX["t_wafer_c"]], ts[:, :, IDX["t_wafer_c"]].max(axis=1),
        z[:, 0], z[:, 1], z[:, 2], z[:, 3], z.max(axis=1) - z.min(axis=1),
        st[:, IDX["t_cool_c"]], st[:, IDX["rf_power_w"]], st[:, IDX["ch_press_mtorr"]]])
    return out.astype(np.float32)
