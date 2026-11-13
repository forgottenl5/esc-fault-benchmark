"""Per-wafer summaries for v2 (v1 summaries plus peak leakage current and minimum clamp-voltage readback)."""
import numpy as np
from .sim2 import CHANNELS

IDX = {c: i for i, c in enumerate(CHANNELS)}
STEADY = slice(15, 78)
SUMMARY_COLS = ["v_rb_mean", "i_leak_mean", "he_p_set_mean", "he_p_mean", "he_flow_mean",
                "t_wafer_mean", "t_wafer_max", "t_z1_mean", "t_z2_mean", "t_z3_mean", "t_z4_mean",
                "t_zone_spread", "t_cool_mean", "rf_power_mean", "ch_press_mean",
                "i_leak_max", "v_rb_min_frac", "ch_press_max"]

# sensor resolution used when writing the data (assumption; typical logged precision)
RESOLUTION = {"v_set": 0.1, "v_rb": 0.1, "i_leak_ma": 0.0005, "he_p_set_torr": 0.01, "he_p_torr": 0.01,
              "he_flow_sccm": 0.01, "t_wafer_c": 0.01, "t_z1_c": 0.01, "t_z2_c": 0.01, "t_z3_c": 0.01,
              "t_z4_c": 0.01, "t_cool_c": 0.01, "rf_power_w": 0.1, "ch_press_mtorr": 0.01}


def quantize(ts):
    out = ts.copy()
    for c, r in RESOLUTION.items():
        k = IDX[c]
        out[:, :, k] = np.round(out[:, :, k] / r) * r
    return out.astype(np.float32)


def summarize(ts):
    st = ts[:, STEADY, :]
    m = st.mean(axis=1)
    z = m[:, IDX["t_z1_c"]:IDX["t_z4_c"] + 1]
    vset = np.maximum(st[:, :, IDX["v_set"]], 1.0)
    out = np.column_stack([
        m[:, IDX["v_rb"]], m[:, IDX["i_leak_ma"]], m[:, IDX["he_p_set_torr"]], m[:, IDX["he_p_torr"]],
        m[:, IDX["he_flow_sccm"]], m[:, IDX["t_wafer_c"]], ts[:, :, IDX["t_wafer_c"]].max(axis=1),
        z[:, 0], z[:, 1], z[:, 2], z[:, 3], z.max(axis=1) - z.min(axis=1),
        m[:, IDX["t_cool_c"]], m[:, IDX["rf_power_w"]], m[:, IDX["ch_press_mtorr"]],
        st[:, :, IDX["i_leak_ma"]].max(axis=1), (st[:, :, IDX["v_rb"]] / vset).min(axis=1),
        st[:, :, IDX["ch_press_mtorr"]].max(axis=1)])
    return out.astype(np.float32)
