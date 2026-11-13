# Dataset card: ESC-Fault v1

## Summary

Simulated sensor data from 140 electrostatic chucks (98 train, 14 validation, 28 test), each run for
550 to 1,200 wafers, about 125,000 wafers in total. Classes are balanced by chuck: 20 chucks per class.
Splits are by chuck, so no chuck appears in two splits.

## How the data is made

`escbench/sim.py` is a lumped model with five linked parts:

1. **Electrical.** Leakage current = clamp voltage / chuck resistance. Resistance falls with temperature
   and falls further as a conductive film builds up.
2. **Clamping.** Clamping force falls as leakage rises. Weaker clamping lets more helium escape.
3. **Thermal.** RF heat into the wafer is removed through the helium gap and through direct contact. A
   run-to-run controller sets the helium pressure that holds the wafer at its target temperature.
4. **Helium.** Flow = leak path x pressure, capped at the supply limit. Escaping helium raises chamber pressure.
5. **Release.** Leftover charge decays; release time is how long it takes to fall below a safe level.

Every chuck gets its own parameters, three recipes in random order, plasma cleans every 200 to 320
wafers, post-clean seasoning, 0 to 3 small sensor recalibrations, slow coolant drift, slow normal wear,
and wafer-to-wafer variation.

## Sourced values and assumptions

| Quantity | Value used | Status |
| --- | --- | --- |
| Clamp voltage range | 600 to 1000 V | Consistent with 0 to 1000 V supply in US5886865 and 2000 V in MDPI Electronics 11(6):880 |
| Nominal leakage current | about 0.05 to 0.1 mA | Sourced: about 0.1 mA nominal, US5886865 |
| Resistance vs temperature | 10x drop from 20 to 150 C | Sourced: resistivity about 1e14 to 1e13 ohm-cm, US5886865 |
| Contamination end of life | leakage about 2x normal | Sourced: alarm at about 2x nominal, US5886865 |
| Wear mechanism and end of life | pressure falls toward about 5 Torr | Sourced: US7582491 (backside gas pressure trend, about 5 Torr lower limit) |
| Heater zone fault size | up to about 6 C | Sourced scale: +/-5.6 C deviation on a bad chuck, MDPI Electronics 11(6):880 |
| Higher voltage lengthens release | yes | Sourced: MDPI Electronics 11(6):880 |
| Helium-cooling leak fault type | yes | Sourced: flowcool leak fault in the PHM 2018 Data Challenge |
| Helium flow (2 to 12 sccm), supply cap 12 sccm | | Assumption |
| Gas conduction curve, contact conductance, RF absorbed fraction (30%) | | Assumption |
| Recipes, chamber pressures, release threshold, residual-charge growth | | Assumption |
| Fault sizes, onset ranges, nuisance events, noise levels | | Assumption, tuned so baselines are neither perfect nor useless |

Sources:
- [US5886865, Method and apparatus for predicting failure of an electrostatic chuck](https://patents.google.com/patent/US5886865)
- [US7582491, Method for diagnosing electrostatic chuck, vacuum processing apparatus, and storage medium](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7582491)
- [On-wafer temperature monitoring for repaired electrostatic chucks, MDPI Electronics 11(6):880](https://mdpi.com/2079-9292/11/6/880)
- [PHM Society 2018 Data Challenge](https://phmsociety.org/wp-content/uploads/2018/05/PHM-Data-Challenge-2018-vFinal-v2_0.pdf)

## Labels

- `fault`: chuck-level class.
- `severity`: 0 before onset, rising to 1. Run-to-failure chucks (contamination, wear) end when severity reaches 1.
- `rul`: wafers until end of life (run-to-failure chucks only, else -1).
- Fault size varies per chuck (`magnitude` 0.2 to 1.0 in the chuck metadata), so some faults stay small.
- Every fault starts at wafer 150 or later; wafers 0 to 99 are always healthy.

## Reproducibility

Train and validation rebuild exactly from the public seed in `generate.py`
(`tests/test_reproducible.py` checks this). Test chucks use a private seed, so test answers cannot be
rebuilt from the code.

## Intended use

Benchmarking fault detection, diagnosis and remaining-life methods on equipment-like multivariate data;
teaching; testing whether a method separates sensor faults from equipment faults.

## Limitations

- Simulated, simplified physics. Real chucks have effects not modeled here: arcing events, RF coupling,
  wafer bow, edge rings, multi-fault interactions, aging of the power supply and more.
- Only one fault per chuck.
- Model results do not transfer to real tools without validation on real data.

## Provenance and ethics

No real fab data, no company-confidential information. Physical behaviour is taken from public patents,
papers and datasets only.

---

# Dataset card: ESC-Fault v2

v2 is a separate release; v1 above is unchanged and stays comparable. Code: `escbench/sim2.py`,
`escbench/features_v2.py`, `escbench/tasks_v2.py`, `generate2.py`, `make_tasks2.py`, `baselines2.py`, `score2.py`.

## What changed from v1

| Change | Why |
| --- | --- |
| 4x more chucks: 40 per class in train (v1: 14), 10 per class in test (v1: 4) | v1's hard classes were partly a small-sample effect; per-class test scores were noisy |
| New fault: `arcing` (micro-arcs: short leakage-current spikes, clamp-voltage dips, pressure blips), with rare background arcs on every chuck | Arcing / voltage breakdown is a known ESC failure mode (see US9417280) |
| Overlapping faults: 8 fault pairs (e.g. contamination + sensor drift, helium leak + heater zone) | Real chucks can have more than one problem; task 1 becomes multi-label |
| Intermittent helium leaks (30% of leaks) and slow onsets (30% of non-wear-out faults) | Harder, more realistic early detection |
| A new recipe appears mid-life on 30% of chucks, never seen in the reference wafers | Methods must cope with a recipe they have no baseline for |
| `test_shift` split from a second tool profile (B): other recipes, more leakage, lower gas conduction, 30% more sensor noise | Tests transfer to another tool, which fabs need and fixed datasets cannot test |
| Signals rounded to a realistic logging resolution; 3 new per-wafer summaries (peak leakage current, minimum clamp-voltage readback ratio, peak chamber pressure) | Smaller files; arcing needs peak values, not just means |
| Generator options (`--per-class`, `--seed`, `--raw`) | Anyone can make more training data and study how much data a method needs |

## Splits

| Split | Tool | Chucks | Wafers | Labels |
| --- | --- | --- | --- | --- |
| train | A | 360 (40 per class x 8 classes, 5 per fault pair x 8 pairs) | 319,205 | public |
| val | A | 48 | 42,030 | public |
| test | A | 96 (10 per class, 2 per pair) | 85,351 | held out (private seed) |
| test_shift | B | 48 (5 per class, 1 per pair) | 43,100 | held out (private seed) |

Train and validation rebuild exactly from the public seed (`tests/test_reproducible_v2.py`). Raw 1 Hz time
series are published for test and test_shift; for train and validation, rebuild them with
`python generate2.py --splits train,val --raw`.

## Labels

Per wafer: severity per fault (`sev_<fault>`, 0 to 1), `label` (faults with severity at least 0.15 joined
by `+`, or `healthy`) and `rul` (wafers to end of life for chucks with contamination or wear, else -1).
Task-1 windows where any fault is between 0 and 0.15 severity are skipped as ambiguous.

## v2 assumptions (in addition to v1)

All new constants are assumptions, chosen so the faults are detectable but not trivial: arc rate (0.004 per
wafer background, up to 0.8 per wafer at full severity), spike size (0.15 to 0.6 mA for 1 to 2 s), voltage dip
(2 to 6%), pressure blip (1 to 4 mTorr); tool B recipes and hardware factors; the mid-life recipe; intermittent
leak episode lengths (10 to 60 wafers on, 20 to 100 off); slow-onset factor (2.5x); logging resolution.

## Known limits of v2

- Still a lumped, simulated model, not validated on real tool data.
- Chuck length differs between wear-out chucks (end at failure) and others, as in v1; methods should not use
  the total length, which is only known in hindsight (all tasks are causal).
- Tool B is one alternative parameter set, not a model of a specific real tool.
