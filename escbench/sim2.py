"""ESC-Fault v2 simulator.

Extends the v1 simulator (escbench/sim.py, kept frozen) with:
  - a second tool profile ("B": different recipes, hardware constants and sensor noise) for tool-to-tool tests
  - an optional new recipe that appears mid-life (not present in the chuck's reference wafers)
  - a new fault, `arcing` (micro-arcs: short leakage-current spikes, clamp-voltage dips, pressure blips),
    plus rare background arcs on every chuck
  - two faults at once on some chucks (overlapping faults), with a severity curve per fault
  - intermittent helium leaks and slow, gradual onsets
  - extra per-wafer summaries (peak leakage current, minimum clamp-voltage readback) in escbench/features_v2.py

Lumped, simplified physics; see DATASET_CARD.md (v2 section) for sourced values and assumptions.
"""
import numpy as np

T = 90
TIME = np.arange(T, dtype=np.float64)
CHANNELS = ["v_set", "v_rb", "i_leak_ma", "he_p_set_torr", "he_p_torr", "he_flow_sccm",
            "t_wafer_c", "t_z1_c", "t_z2_c", "t_z3_c", "t_z4_c", "t_cool_c",
            "rf_power_w", "ch_press_mtorr"]
FAULTS = ["contamination", "wear", "he_leak", "residual_charge", "heater_zone", "arcing", "sensor_drift"]
RUN_TO_FAILURE = ("contamination", "wear")

TOOLS = {
    "A": dict(recipes=[dict(v=600.0, rf=800.0, tw=70.0, tc=40.0, pch=20.0),
                       dict(v=800.0, rf=1200.0, tw=85.0, tc=45.0, pch=30.0),
                       dict(v=1000.0, rf=1500.0, tw=100.0, tc=50.0, pch=40.0)],
              new_recipe=dict(v=900.0, rf=1000.0, tw=80.0, tc=45.0, pch=25.0),
              hg=1.0, l0=1.0, r_ref=1.0, noise=1.0, zone_sd=0.5),
    "B": dict(recipes=[dict(v=700.0, rf=900.0, tw=75.0, tc=42.0, pch=15.0),
                       dict(v=900.0, rf=1300.0, tw=90.0, tc=48.0, pch=25.0),
                       dict(v=1100.0, rf=1700.0, tw=105.0, tc=55.0, pch=35.0)],
              new_recipe=dict(v=750.0, rf=1100.0, tw=82.0, tc=44.0, pch=20.0),
              hg=0.9, l0=1.3, r_ref=0.7, noise=1.3, zone_sd=0.8),
}

V_ON, HE_ON, RF_ON, RF_OFF, HE_OFF, V_OFF = 2, 5, 10, 80, 82, 85
ABS_FRAC = 0.30
P0 = 8.0
HE_SUPPLY_MAX = 12.0
B_RES = 2197.0
T_REF_K = 333.15
V_TH = 20.0


def _curve(rng, f, n, onset, dur):
    """Severity 0..1 for one fault."""
    x = np.clip((n - onset) / dur, 0, None)
    if f == "contamination":
        s = x ** 1.5
    elif f in ("wear", "residual_charge", "sensor_drift", "arcing"):
        s = x
    else:  # he_leak, heater_zone
        if rng.random() < 0.5:
            s = np.where(n >= onset, rng.uniform(0.35, 0.8) + 0.2 * x, 0.0)
        else:
            s = x * (dur / rng.uniform(30, 90))
    s = np.clip(s, 0, 1)
    if f == "he_leak" and rng.random() < 0.3:
        # intermittent leak: the seal leaks in episodes, then reseats
        mask = np.zeros(len(n)); i = int(onset)
        while i < len(n):
            on = int(rng.integers(10, 60)); off = int(rng.integers(20, 100))
            mask[i:i + on] = 1.0; i += on + off
        s = s * mask
    return s


def simulate_chuck(rng, faults, chuck_id, tool="A", new_recipe_at=None):
    """faults: list of 0, 1 or 2 fault names. Returns dict(ts, wafer, meta)."""
    faults = list(faults)
    assert all(f in FAULTS for f in faults) and len(faults) <= 2
    assert sum(f in RUN_TO_FAILURE for f in faults) <= 1
    P = TOOLS[tool]
    r_ref = np.exp(rng.normal(np.log(7.0e6 * P["r_ref"]), 0.15))
    hg_max = 11.7 * P["hg"] * rng.normal(1, 0.05)
    hc0 = 1.5 * rng.normal(1, 0.10)
    l0 = 0.25 * P["l0"] * rng.normal(1, 0.10)
    tau_r = 1.0 * rng.normal(1, 0.10)
    zone_off = rng.normal(0, P["zone_sd"], 4)
    t_cool = rng.normal(20.0, 0.5)
    contam_rate = rng.uniform(0.0002, 0.0005)
    clean_every = int(rng.integers(200, 320))
    nz = P["noise"]

    rtf = [f for f in faults if f in RUN_TO_FAILURE]
    plan = {}
    if rtf:
        onset = int(rng.integers(150, 400)); dur = int(rng.integers(400, 800)); N = onset + dur
        plan[rtf[0]] = (onset, dur)
    else:
        N = int(rng.integers(700, 1100))
    for f in faults:
        if f in plan:
            continue
        onset = int(rng.integers(150, max(151, min(600, N - 150))))
        dur = int(rng.integers(250, 600))
        if rng.random() < 0.3:
            dur = int(dur * 2.5)          # slow, gradual onset
        plan[f] = (onset, dur)
    n = np.arange(N)
    sev = {f: _curve(rng, f, n, *plan[f]) for f in faults}
    if rtf:
        sev[rtf[0]][-1] = 1.0
    mag = {f: (1.0 if f in RUN_TO_FAILURE else rng.uniform(0.2, 1.0)) for f in faults}
    S = lambda f: sev[f] if f in sev else np.zeros(N)
    M = lambda f: mag.get(f, 0.0)

    s_bg_wear = rng.uniform(0.0, 0.2) * n / 1000.0
    wv_flow = rng.lognormal(0, 0.06 * nz, N)
    wv_temp = rng.normal(0, 1.0 * nz, N)
    wv_leak = rng.lognormal(0, 0.05 * nz, N)
    wv_pch = rng.normal(0, 0.4 * nz, N)
    wv_zone = rng.normal(0, 0.25 * nz, (N, 4))
    cool_drift = 0.4 * np.sin(2 * np.pi * n / rng.uniform(250, 450) + rng.uniform(0, 6.3))

    c_norm = np.zeros(N); c = 0.0; since_clean = np.zeros(N, dtype=np.int32); k = 0
    for i in range(N):
        if i > 0 and i % clean_every == 0:
            c *= 0.2; k = 0
        c += contam_rate * rng.uniform(0.5, 1.5)
        c_norm[i] = c; since_clean[i] = k; k += 1
    c_fault = 1.0 * S("contamination")
    season = np.exp(-since_clean / 8.0) * (n >= clean_every)
    calib = []
    for _ in range(int(rng.integers(0, 4))):
        calib.append((int(rng.integers(50, N)), str(rng.choice(["i", "flow", "zone", "pch"])), float(rng.normal(0, 1))))
    c_tot = c_norm + c_fault

    recipes = list(P["recipes"])
    recipe = rng.choice(3, size=N, p=[0.4, 0.35, 0.25])
    if new_recipe_at is not None and new_recipe_at < N:
        recipes.append(P["new_recipe"])
        late = (n >= new_recipe_at) & (rng.random(N) < 0.25)
        recipe = np.where(late, 3, recipe)
    V = np.array([recipes[r]["v"] for r in recipe])
    RF = np.array([recipes[r]["rf"] for r in recipe]) * rng.normal(1, 0.005, N)
    TW = np.array([recipes[r]["tw"] for r in recipe])
    TC = np.array([recipes[r]["tc"] for r in recipe])
    PCH = np.array([recipes[r]["pch"] for r in recipe])

    zone_true = np.zeros((N, 4))
    bad_zone = int(rng.integers(0, 4))
    if "heater_zone" in faults:
        zone_true[:, bad_zone] = rng.choice([-1, 1]) * 6.0 * M("heater_zone") * S("heater_zone")
    t_chuck = TC + zone_true.mean(axis=1) + cool_drift

    f_rel = (V / 800.0) ** 2 / (1.0 + c_tot)
    f_ref = (V / 800.0) ** 2
    l_path = l0 * (1.0 + 0.8 * (f_ref / f_rel - 1.0))
    l_path = l_path + 1.2 * M("he_leak") * S("he_leak")
    l_path = l_path * wv_flow * (1 + 0.08 * season)
    hc = hc0 * (1.0 + 1.33 * S("wear") + 1.33 * s_bg_wear)

    pabs = ABS_FRAC * RF
    h_need = pabs / np.maximum(TW - t_chuck, 5.0) - hc
    frac = np.clip(h_need / hg_max, 0.02, 0.95)
    p_req = P0 * frac / (1 - frac)
    p_set = np.empty(N); mem = {}
    for i in range(N):
        r = int(recipe[i])
        mem[r] = p_req[i] if r not in mem else mem[r] + 0.5 * (p_req[i] - mem[r])
        p_set[i] = mem[r]
    p_set = np.clip(p_set + rng.normal(0, 0.05, N), 1.0, 20.0)
    flow_want = l_path * p_set
    p_act = np.where(flow_want > HE_SUPPLY_MAX, HE_SUPPLY_MAX / l_path, p_set)
    flow = np.minimum(flow_want, HE_SUPPLY_MAX)

    r_fac = 8.0 * M("residual_charge") * S("residual_charge")
    dechuck = tau_r * np.log(V / V_TH * (1 + r_fac)) * rng.normal(1, 0.04, N)
    p_stick = 0.01 + 0.15 * M("residual_charge") * S("residual_charge")
    stick = rng.random(N) < p_stick
    dechuck = dechuck + stick * rng.uniform(2, 6, N)

    t = TIME[None, :]
    v_on = ((t >= V_ON) & (t < V_OFF)).astype(float)
    he_ramp = np.clip((t - HE_ON) / 3.0, 0, 1) * (t < HE_OFF)
    rf_on = ((t >= RF_ON) & (t < RF_OFF)).astype(float)
    heat = np.where(t < RF_ON, 0.0, np.where(t < RF_OFF, 1 - np.exp(-(t - RF_ON) / 6.0),
                    (1 - np.exp(-(RF_OFF - RF_ON) / 6.0)) * np.exp(-(t - RF_OFF) / 6.0)))

    v_set = V[:, None] * v_on
    v_rb = v_set * (1 + rng.normal(0, 0.003 * nz, (N, T)))
    p_ts_set = p_set[:, None] * he_ramp
    p_ts = p_act[:, None] * he_ramp
    flow_ts = flow[:, None] * he_ramp
    h_gas = hg_max * p_act / (p_act + P0)
    dT = pabs / (h_gas + hc)
    t_wafer = t_chuck[:, None] + dT[:, None] * heat + wv_temp[:, None]
    chuck_rise = 2.0 * heat
    t_ck = t_chuck[:, None] + chuck_rise
    R = r_ref * np.exp(B_RES * (1.0 / (t_ck + 273.15) - 1.0 / T_REF_K))
    i_leak = v_set / R * (1.0 + c_tot[:, None]) * (wv_leak * (1 - 0.05 * season))[:, None] * 1e3
    i_leak = i_leak + (t == V_ON) * 0.3 * V[:, None] / 800.0
    zones = (TC[:, None, None] + cool_drift[:, None, None] + zone_off[None, None, :] + zone_true[:, None, :] + wv_zone[:, None, :]
             + chuck_rise[:, :, None] * np.array([1.0, 1.0, 1.0, 1.3]))
    ch_press = PCH[:, None] * (0.3 + 0.7 * rf_on) + 0.15 * flow_ts + wv_pch[:, None] * rf_on
    rf_ts = RF[:, None] * rf_on

    # ---- micro-arcs: rare on every chuck, frequent with the arcing fault (assumption) ----
    arc_rate = 0.004 + 0.8 * M("arcing") * S("arcing")      # expected arcs per wafer
    n_arcs = rng.poisson(arc_rate)
    arc_i = np.zeros((N, T)); arc_v = np.zeros((N, T)); arc_p = np.zeros((N, T))
    for w in np.nonzero(n_arcs)[0]:
        for _ in range(int(n_arcs[w])):
            a = int(rng.integers(RF_ON + 2, RF_OFF - 3)); d = int(rng.integers(1, 3))
            arc_i[w, a:a + d] += rng.uniform(0.15, 0.6)           # mA spike
            arc_v[w, a] += rng.uniform(0.02, 0.06)                # fractional clamp-voltage dip
            arc_p[w, a:a + 2] += rng.uniform(1.0, 4.0)            # mTorr blip
    i_leak = i_leak + arc_i * v_on
    v_rb = v_rb * (1 - arc_v)
    ch_press = ch_press + arc_p

    ts = np.zeros((N, T, len(CHANNELS)))
    ts[:, :, 0] = v_set
    ts[:, :, 1] = v_rb
    ts[:, :, 2] = i_leak + rng.normal(0, 0.004 * nz, (N, T))
    ts[:, :, 3] = p_ts_set
    ts[:, :, 4] = p_ts + rng.normal(0, 0.05 * nz, (N, T))
    ts[:, :, 5] = flow_ts + rng.normal(0, 0.08 * nz, (N, T))
    ts[:, :, 6] = t_wafer + rng.normal(0, 0.4 * nz, (N, T))
    ts[:, :, 7:11] = zones + rng.normal(0, 0.2 * nz, (N, T, 4))
    ts[:, :, 11] = t_cool + cool_drift[:, None] + rng.normal(0, 0.1 * nz, (N, T))
    ts[:, :, 12] = rf_ts * (1 + rng.normal(0, 0.002 * nz, (N, T)))
    ts[:, :, 13] = ch_press + rng.normal(0, 0.3 * nz, (N, T))

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

    drift_sensor = ""
    if "sensor_drift" in faults:
        s = S("sensor_drift"); mg = M("sensor_drift")
        drift_sensor = str(rng.choice(["i_leak_ma", "he_flow_sccm", "t_zone", "he_p_torr"]))
        if drift_sensor == "i_leak_ma":
            ts[:, :, 2] *= (1 + 1.0 * mg * s)[:, None]
        elif drift_sensor == "he_flow_sccm":
            ts[:, :, 5] += (4.0 * mg * s)[:, None] * (he_ramp > 0)
        elif drift_sensor == "he_p_torr":
            ts[:, :, 4] *= (1 - 0.3 * mg * s)[:, None]
        else:
            zz = int(rng.integers(0, 4))
            ts[:, :, 7 + zz] += (rng.choice([-1, 1]) * 5.0 * mg * s)[:, None]
            drift_sensor = f"t_z{zz + 1}_c"

    rul = (N - 1 - n).astype(np.int32) if rtf else np.full(N, -1, np.int32)
    wafer = dict(wafer_idx=n.astype(np.int32), recipe=recipe.astype(np.int8),
                 dechuck_time_s=dechuck.astype(np.float32),
                 wafers_since_install=n.astype(np.int32), wafers_since_clean=since_clean,
                 rul=rul, **{f"sev_{f}": S(f).astype(np.float32) for f in FAULTS})
    meta = dict(chuck_id=int(chuck_id), faults=faults, tool=tool,
                new_recipe_at=int(new_recipe_at) if new_recipe_at is not None and new_recipe_at < N else -1,
                onsets={f: int(plan[f][0]) for f in faults}, n_wafers=int(N), run_to_failure=bool(rtf),
                drift_sensor=drift_sensor, bad_zone=bad_zone + 1 if "heater_zone" in faults else -1,
                magnitudes={f: float(mag[f]) for f in faults})
    return dict(ts=ts.astype(np.float32), wafer=wafer, meta=meta)
