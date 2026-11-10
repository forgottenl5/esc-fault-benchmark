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
