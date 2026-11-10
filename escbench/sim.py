"""ESC (electrostatic chuck) fault simulator.

Generates per-wafer time series for one chuck over its life, with optional fault.
Lumped, simplified physics; see DATASET_CARD.md for which constants are sourced
and which are assumptions.
"""
import numpy as np

T = 90                      # seconds per wafer, sampled at 1 Hz
TIME = np.arange(T, dtype=np.float64)
CHANNELS = ["v_set", "v_rb", "i_leak_ma", "he_p_set_torr", "he_p_torr", "he_flow_sccm",
            "t_wafer_c", "t_z1_c", "t_z2_c", "t_z3_c", "t_z4_c", "t_cool_c",
            "rf_power_w", "ch_press_mtorr"]
FAULTS = ["healthy", "contamination", "wear", "he_leak", "residual_charge",
          "heater_zone", "sensor_drift"]

# recipe: clamp voltage (V), RF power (W), wafer temperature target (C), chuck setpoint (C), base chamber pressure (mTorr)
RECIPES = [dict(v=600.0, rf=800.0, tw=70.0, tc=40.0, pch=20.0),
           dict(v=800.0, rf=1200.0, tw=85.0, tc=45.0, pch=30.0),
           dict(v=1000.0, rf=1500.0, tw=100.0, tc=50.0, pch=40.0)]

# timeline within a wafer (s)
V_ON, HE_ON, RF_ON, RF_OFF, HE_OFF, V_OFF = 2, 5, 10, 80, 82, 85
ABS_FRAC = 0.30             # fraction of RF power absorbed as wafer heat (assumption)
P0 = 8.0                    # Torr, gas-conduction half-saturation pressure (assumption)
HE_SUPPLY_MAX = 12.0        # sccm, backside helium supply limit (assumption)
B_RES = 2197.0              # K; resistivity falls 10x from 20 C to 150 C (US5886865)
T_REF_K = 333.15            # 60 C reference for chuck resistance
V_TH = 20.0                 # V, residual-charge release threshold (assumption)
EOL_P_TORR = 5.0            # wear end-of-life: recipe-0 pressure reaches 5 Torr (US7582491)


def _severity(rng, ftype, n, onset, dur):
    """Severity 0..1 per wafer index n (array)."""
    x = np.clip((n - onset) / dur, 0, None)
    if ftype == "contamination":
        s = x ** 1.5
    elif ftype in ("wear", "residual_charge", "sensor_drift"):
        s = x
    elif ftype in ("he_leak", "heater_zone"):
        if rng.random() < 0.5:   # step change, then slow growth
            s = np.where(n >= onset, rng.uniform(0.35, 0.8) + 0.2 * x, 0.0)
        else:                    # fast ramp
            s = x * (dur / rng.uniform(30, 90))
    else:
        s = np.zeros_like(x)
    return np.clip(s, 0, 1)


def simulate_chuck(rng, ftype, chuck_id):
    """Return dict with 'ts' (N,T,C float32), 'wafer' (dict of per-wafer arrays), 'meta'."""
    assert ftype in FAULTS
    # ---- per-chuck parameters ----
    r_ref = np.exp(rng.normal(np.log(7.0e6), 0.15))      # ohm at 60 C -> ~0.1 mA at 700 V (US5886865 nominal)
    hg_max = 11.7 * rng.normal(1, 0.05)                  # W/K gas conduction at saturation
    hc0 = 1.5 * rng.normal(1, 0.10)                      # W/K direct contact conduction
    l0 = 0.25 * rng.normal(1, 0.10)                      # sccm/Torr baseline backside leak path
    tau_r = 1.0 * rng.normal(1, 0.10)                    # s, residual charge decay
    zone_off = rng.normal(0, 0.5, 4)                     # C, per-zone calibration offsets
    t_cool = rng.normal(20.0, 0.5)                       # C coolant
    contam_rate = rng.uniform(0.0002, 0.0005)            # normal slow film buildup per wafer
    clean_every = int(rng.integers(200, 320))

    run_to_failure = ftype in ("contamination", "wear")
    if run_to_failure:
        onset = int(rng.integers(150, 400)); dur = int(rng.integers(400, 800))
        N = onset + dur
    else:
        N = int(rng.integers(700, 1100))
        onset = int(rng.integers(150, 600)) if ftype != "healthy" else N
        dur = int(rng.integers(250, 600))
    n = np.arange(N)
    s = _severity(rng, ftype, n, onset, dur) if ftype != "healthy" else np.zeros(N)
    if run_to_failure:
        s[-1] = 1.0
    # size of the fault at full severity, for faults that do not end the chuck's life (assumption)
    mag = 1.0 if run_to_failure else rng.uniform(0.2, 1.0)
    # every chuck wears a little in normal use (assumption), so a falling pressure is not unique to the wear fault
    s_bg_wear = rng.uniform(0.0, 0.2) * n / 1000.0
    # wafer-to-wafer variation (backside condition, emissivity, delivered RF) (assumption)
    wv_flow = rng.lognormal(0, 0.06, N)
    wv_temp = rng.normal(0, 1.0, N)
    wv_leak = rng.lognormal(0, 0.05, N)
    wv_pch = rng.normal(0, 0.4, N)
    wv_zone = rng.normal(0, 0.25, (N, 4))
    cool_drift = 0.4 * np.sin(2 * np.pi * n / rng.uniform(250, 450) + rng.uniform(0, 6.3))

    # ---- maintenance: plasma cleans partly remove normal film ----
    c_norm = np.zeros(N); c = 0.0; since_clean = np.zeros(N, dtype=np.int32); k = 0
    for i in range(N):
        if i > 0 and i % clean_every == 0:
            c *= 0.2; k = 0
        c += contam_rate * rng.uniform(0.5, 1.5)
        c_norm[i] = c; since_clean[i] = k; k += 1
    c_fault = 1.0 * s if ftype == "contamination" else np.zeros(N)
    # after a clean the surface 'seasons' for ~20 wafers: helium flow a little higher, current a little lower (assumption)
    season = np.exp(-since_clean / 8.0) * (n >= clean_every)
    # 0-3 sensor recalibrations per chuck: small permanent steps in one reading; normal events, not faults (assumption)
    calib = []
    for _ in range(int(rng.integers(0, 4))):
        calib.append((int(rng.integers(50, N)), str(rng.choice(["i", "flow", "zone", "pch"])), float(rng.normal(0, 1))))
    c_tot = c_norm + c_fault

    recipe = rng.choice(3, size=N, p=[0.4, 0.35, 0.25])
    V = np.array([RECIPES[r]["v"] for r in recipe])
    RF = np.array([RECIPES[r]["rf"] for r in recipe]) * rng.normal(1, 0.005, N)
    TW = np.array([RECIPES[r]["tw"] for r in recipe])
    TC = np.array([RECIPES[r]["tc"] for r in recipe])
    PCH = np.array([RECIPES[r]["pch"] for r in recipe])

    # ---- heater zone fault: one zone's true temperature offset ----
    zone_true = np.tile(zone_off, (N, 1)) * 0.0
    bad_zone = int(rng.integers(0, 4))
    if ftype == "heater_zone":
        sign = rng.choice([-1, 1])
        zone_true[:, bad_zone] = sign * 6.0 * mag * s
    t_chuck = TC + zone_true.mean(axis=1) + cool_drift                 # mean true chuck surface temperature

    # ---- clamping and helium ----
    f_rel = (V / 800.0) ** 2 / (1.0 + c_tot)             # clamp force falls as leakage rises (US5886865)
    f_ref = (V / 800.0) ** 2
    l_path = l0 * (1.0 + 0.8 * (f_ref / f_rel - 1.0))
    if ftype == "he_leak":
        l_path = l_path + 1.2 * mag * s
    l_path = l_path * wv_flow * (1 + 0.08 * season)
    hc = hc0 * (1.0 + 1.33 * (s if ftype == "wear" else 0) + 1.33 * s_bg_wear)

    # pressure required to hold wafer target temperature; run-to-run controller follows it per recipe
    pabs = ABS_FRAC * RF
    h_need = pabs / np.maximum(TW - t_chuck, 5.0) - hc
    frac = np.clip(h_need / hg_max, 0.02, 0.95)
    p_req = P0 * frac / (1 - frac)
    p_set = np.empty(N); mem = {0: None, 1: None, 2: None}
    for i in range(N):
        r = recipe[i]
        mem[r] = p_req[i] if mem[r] is None else mem[r] + 0.5 * (p_req[i] - mem[r])
        p_set[i] = mem[r]
    p_set = np.clip(p_set + rng.normal(0, 0.05, N), 1.0, 20.0)
    flow_want = l_path * p_set
    p_act = np.where(flow_want > HE_SUPPLY_MAX, HE_SUPPLY_MAX / l_path, p_set)
    flow = np.minimum(flow_want, HE_SUPPLY_MAX)

    # ---- dechuck (release) time ----
    r_fac = 8.0 * mag * s if ftype == "residual_charge" else np.zeros(N)
    dechuck = tau_r * np.log(V / V_TH * (1 + r_fac)) * rng.normal(1, 0.04, N)
    p_stick = 0.01 + (0.15 * mag * s if ftype == "residual_charge" else 0)
    stick = rng.random(N) < p_stick
    dechuck = dechuck + stick * rng.uniform(2, 6, N)

    # ---- time series ----
    t = TIME[None, :]
    v_on = ((t >= V_ON) & (t < V_OFF)).astype(float)
    he_ramp = np.clip((t - HE_ON) / 3.0, 0, 1) * (t < HE_OFF)
    rf_on = ((t >= RF_ON) & (t < RF_OFF)).astype(float)
    heat = np.where(t < RF_ON, 0.0, np.where(t < RF_OFF, 1 - np.exp(-(t - RF_ON) / 6.0),
                    (1 - np.exp(-(RF_OFF - RF_ON) / 6.0)) * np.exp(-(t - RF_OFF) / 6.0)))

    v_set = V[:, None] * v_on
    v_rb = v_set * (1 + rng.normal(0, 0.003, (N, T)))
    p_ts_set = p_set[:, None] * he_ramp
    p_ts = p_act[:, None] * he_ramp
    flow_ts = flow[:, None] * he_ramp
    h_gas = hg_max * p_act / (p_act + P0)
    dT = pabs / (h_gas + hc)
    t_wafer = t_chuck[:, None] + dT[:, None] * heat + wv_temp[:, None]
    chuck_rise = 2.0 * heat                                # chuck surface warms slightly under RF
    t_ck = t_chuck[:, None] + chuck_rise
    R = r_ref * np.exp(B_RES * (1.0 / (t_ck + 273.15) - 1.0 / T_REF_K))
    i_leak = v_set / R * (1.0 + c_tot[:, None]) * (wv_leak * (1 - 0.05 * season))[:, None] * 1e3      # mA
    i_leak = i_leak + (t == V_ON) * 0.3 * V[:, None] / 800.0  # charging spike
    zones = (TC[:, None, None] + cool_drift[:, None, None] + zone_off[None, None, :] + zone_true[:, None, :] + wv_zone[:, None, :]
             + chuck_rise[:, :, None] * np.array([1.0, 1.0, 1.0, 1.3]))
    ch_press = PCH[:, None] * (0.3 + 0.7 * rf_on) + 0.15 * flow_ts + wv_pch[:, None] * rf_on
    rf_ts = RF[:, None] * rf_on

    ts = np.zeros((N, T, len(CHANNELS)))
    ts[:, :, 0] = v_set
    ts[:, :, 1] = v_rb
    ts[:, :, 2] = i_leak + rng.normal(0, 0.004, (N, T))
    ts[:, :, 3] = p_ts_set
    ts[:, :, 4] = p_ts + rng.normal(0, 0.05, (N, T))
    ts[:, :, 5] = flow_ts + rng.normal(0, 0.08, (N, T))
    ts[:, :, 6] = t_wafer + rng.normal(0, 0.4, (N, T))
    ts[:, :, 7:11] = zones + rng.normal(0, 0.2, (N, T, 4))
    ts[:, :, 11] = t_cool + cool_drift[:, None] + rng.normal(0, 0.1, (N, T))
    ts[:, :, 12] = rf_ts * (1 + rng.normal(0, 0.002, (N, T)))
    ts[:, :, 13] = ch_press + rng.normal(0, 0.3, (N, T))

    for at, kind, z in calib:
        sel = n >= at
        if kind == "i":
            ts[sel, :, 2] *= 1 + 0.08 * z
        elif kind == "flow":
            ts[sel, :, 5] += 0.3 * z * (he_ramp[0] > 0)
        elif kind == "zone":
            ts[sel, :, 7 + int(rng.integers(0, 4))] += 0.6 * z
        else:
            ts[sel, :, 13] += 0.5 * z * rf_on[0]

    # ---- sensor drift decoy: one sensor reads wrong, physics unchanged ----
    drift_sensor = ""
    if ftype == "sensor_drift":
        drift_sensor = rng.choice(["i_leak_ma", "he_flow_sccm", "t_zone", "he_p_torr"])
        if drift_sensor == "i_leak_ma":
            ts[:, :, 2] *= (1 + 1.0 * mag * s)[:, None]
        elif drift_sensor == "he_flow_sccm":
            ts[:, :, 5] += (4.0 * mag * s)[:, None] * (he_ramp > 0)
        elif drift_sensor == "he_p_torr":
            ts[:, :, 4] *= (1 - 0.3 * mag * s)[:, None]
        else:
            ts[:, :, 7 + bad_zone] += (rng.choice([-1, 1]) * 5.0 * mag * s)[:, None]
            drift_sensor = f"t_z{bad_zone + 1}_c"

    # remaining useful life (wafers to end of life) for run-to-failure faults
    rul = (N - 1 - n).astype(np.int32) if run_to_failure else np.full(N, -1, np.int32)

    wafer = dict(wafer_idx=n.astype(np.int32), recipe=recipe.astype(np.int8),
                 dechuck_time_s=dechuck.astype(np.float32),
                 wafers_since_install=n.astype(np.int32), wafers_since_clean=since_clean,
                 severity=s.astype(np.float32), rul=rul)
    meta = dict(chuck_id=chuck_id, fault=ftype, onset_wafer=int(onset) if ftype != "healthy" else -1,
                n_wafers=int(N), run_to_failure=run_to_failure, drift_sensor=drift_sensor,
                bad_zone=bad_zone + 1 if ftype == "heater_zone" else -1, magnitude=float(mag))
    return dict(ts=ts.astype(np.float32), wafer=wafer, meta=meta)
