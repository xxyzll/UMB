# Auto Unknown Calibration Progress

Date: 2026-06-01

## Objective

Evaluate the first bootstrap version of Support-Calibrated Adaptive Unknown Scoring, implemented as `--use_auto_unknown_calibration`, as a practical baseline for exceeding FOMO and then PASS.

## Current Run

- Container: `umb_auto_calibration_20260601`
- Status: completed, exited 0
- Output CSV: `run_outputs/paper/owlvit-large-patch14/t1/auto_unknown_calibration_20260601.csv`
- Log: `run_outputs/auto_unknown_calibration_20260601.log`
- Script: `configs/eval_auto_unknown_calibration.sh`

Update on 2026-06-02:

- Aquatic and Aerial completed.
- The original Game profile used `gm`, but Game had no cached GM distribution and spent many hours fitting distributions at runtime.
- The run was stopped after preserving the completed CSV rows.
- Game was changed to the cached `score` distribution path and the script gained `START_DATASET`/`KEEP_RESULT` resume controls.
- The run was restarted from Game with `START_DATASET=Game KEEP_RESULT=true`.
- The resumed run completed Game, Medical, and Surgical.

## Completed Results

| Domain | AutoCal U_AP50 | AutoCal K_AP50 | FOMO Paper U_AP50 | FOMO Paper K_AP50 | Status |
|---|---:|---:|---:|---:|---|
| Aquatic | 18.912 | 51.595 | 18.2 | 50.1 | beats FOMO on U and K |
| Aerial | 11.389 | 40.088 | 6.0 | 25.3 | beats FOMO on U and K |
| Game | 30.636 | 10.479 | 30.4 | 10.7 | beats FOMO on U, slightly below on K |
| Medical | 13.165 | 22.260 | 9.4 | 21.8 | beats FOMO on U and K |
| Surgical | 21.120 | 36.592 | 12.0 | 29.0 | beats FOMO on U and K |
| Mean | 19.044 | 32.203 | 15.2 | 27.38 | beats FOMO mean U and K |

## Aquatic Configuration

| Field | Value |
|---|---|
| fit_method | `score` |
| balance | `0.2` |
| alpha | `-1.0` |
| support_norm | `true` |
| paper_no_obj_zscore | `true` |
| support_norm_mean | `0.17425870895385742` |
| support_norm_std | `0.03226320818066597` |
| support_norm_n | `1404000` |
| calibration_reason | `score_b0.2_no_obj_zscore_support_norm_for_aquatic_recall` |

## Next Order

1. Treat this AutoCal run as the new FOMO-beating baseline.
2. Compare against PASS under the same Task 1 protocol.
3. Identify domains where PASS still wins.
4. Add the next method layer only for those domains, starting with residual/detail-attribute scoring and/or a known-preserving gate for Game.
5. Re-run targeted domains before launching another full 5-domain sweep.
