# ESC-Fault: a simulated benchmark for electrostatic chuck fault diagnosis

A free, open benchmark for diagnosing faults in the **electrostatic chuck (ESC)**, the part of a
semiconductor etch or deposition tool that holds the wafer with static electricity and cools it with
backside helium.

Real equipment data stays inside chipmakers and toolmakers, so public fault benchmarks for fab tools are
rare. This dataset is **simulated** from simplified physics, with every constant either taken from a
public source or marked as an assumption (see [DATASET_CARD.md](DATASET_CARD.md)).

- 140 simulated chucks, about 125,000 wafers, 14 sensor signals at 1 Hz per wafer
- 6 fault types plus healthy, including a **sensor-fault decoy** (the sensor is wrong, the chuck is fine)
- 3 tasks: diagnose, detect early, predict remaining life
- Held-out test labels: the test chucks come from a private seed

![Per-wafer trends for one chuck of each class](figures/fault_trends.png)

## Fault classes

| Fault | Physical cause | Main signature |
| --- | --- | --- |
| `contamination` | Conductive film on the chuck surface | Leakage current rises, clamping weakens, helium flow rises; runs to end of life |
| `wear` | Surface wears smooth, more wafer contact | Helium pressure needed to hold wafer temperature falls; runs to end of life |
| `he_leak` | Seal or seating problem behind the wafer | Helium flow and chamber pressure rise |
| `residual_charge` | Charge stays after voltage is removed | Release (dechuck) time grows, more sticking events |
| `heater_zone` | One heater zone loses control | Zone temperature spread widens; helium setpoint shifts |
| `sensor_drift` | One sensor drifts; the chuck is fine | One reading drifts while related signals stay consistent |

Normal events that are **not** faults also appear in every chuck: plasma cleans, post-clean "seasoning",
small sensor recalibrations, coolant drift, recipe changes and slow normal wear.

## Files

| Path | Contents |
| --- | --- |
| `data_small/wafers_{train,val,test}.parquet` | One row per wafer: recipe, counters, release time and 15 summary features |
| `data_small/labels_{train,val}.parquet` | Per wafer: fault, severity (0 to 1), remaining life |
| `data_small/chucks_{train,val}.json` | Per chuck: fault, onset wafer, length, fault size |
| `data_small/timeseries_sample_train.parquet` | Raw 1 Hz signals for 2 training chucks |
| `tasks/` | Task files for all splits (test files have no answers) |
| Full raw time series (about 500 MB compressed) | [Hugging Face: forgottenl5/esc-fault-benchmark](https://huggingface.co/datasets/forgottenl5/esc-fault-benchmark), or rebuild train/val with `generate.py` |

Raw signals: `v_set, v_rb, i_leak_ma, he_p_set_torr, he_p_torr, he_flow_sccm, t_wafer_c, t_z1_c..t_z4_c,
t_cool_c, rf_power_w, ch_press_mtorr`. Each wafer is 90 s: clamp at 2 s, helium at 5 s, RF from 10 to
80 s, release at 85 s.

## Tasks

All tasks are **causal**: a prediction at wafer *n* may use only wafers 0 to *n* of that chuck. The
first 100 wafers of every chuck are healthy and may be used as its reference.

| Task | Question | Score |
| --- | --- | --- |
| 1. Diagnose | For each (chuck, end wafer): which of the 7 classes? Faulty once severity is at least 0.15 | Macro-F1 |
| 2. Detect early | One alarm per chuck, with the fault type | Detection rate, median delay in wafers, false alarms per 1,000 healthy wafers |
| 3. Remaining life | For chucks that wear out: wafers left after the end wafer | Mean absolute error; asymmetric score that penalizes late (too optimistic) guesses more |

The test task-3 file also contains decoy rows on chucks that do not wear out, so it does not leak
task-1 answers. Decoys are not scored.

## Baseline results (test set, 28 chucks)

| Model | Task 1 macro-F1 | Task 2 detection | Task 2 median delay | Task 2 false alarms / 1k | Task 3 MAE (wafers) |
| --- | --- | --- | --- | --- | --- |
| Threshold rules | 0.667 | 87.5% | 152 | 0.00 | n/a |
| Random forest | **0.812** | 83.3% | **73** | 0.17 | **65.5** |
| Small neural net (MLP) | 0.782 | 83.3% | 74.5 | 0.17 | n/a |
| Guess the mean | n/a | n/a | n/a | n/a | 123.4 |

Per-class task-1 F1 for the random forest: heater_zone 0.57 and sensor_drift 0.23 are the open problems;
the other classes are 0.95 or higher. Telling a failing sensor from a failing chuck is the hard part.

## Quick start

```bash
pip install -r requirements.txt
python generate.py --out data --splits train,val     # rebuild train/val exactly (about 15 s)
python make_tasks.py --data data --out tasks --splits train,val
python baselines.py --data data --tasks tasks --eval val
python tests/test_reproducible.py                     # checks the rebuild matches the published files
```

The per-wafer tables in `data_small/` are enough for all three tasks. Raw time series are optional.

## Submitting test results

Run your model on the test task files and open an issue with three CSVs (formats in `score.py`):
`task1: chuck_id,end_wafer,label`, `task2: chuck_id,alarm_wafer,fault`, `task3: chuck_id,end_wafer,rul_pred`.
Results will be scored against the held-out answers and added to a leaderboard here.

## Limits

This is simulated data from a lumped model, not measurements from a real tool. Several constants are
assumptions, listed in the dataset card. Good scores here show a method can separate these fault
patterns; they do not prove it will work on a real chuck. Corrections from people who work with ESCs are
welcome; please open an issue.

## License

Code: MIT. Data: CC BY 4.0. Author: forgottenl5.
