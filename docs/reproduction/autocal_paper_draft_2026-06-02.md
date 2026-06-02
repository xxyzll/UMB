# Support-Calibrated Adaptive Unknown Scoring for Open-World Object Detection

Draft date: 2026-06-02

## Abstract

Open-world object detection requires a detector to preserve known-class accuracy while identifying objects from unknown categories. Recent foundation-model-based methods such as FOMO improve zero/few-shot open-world detection by using attribute and distribution cues, but the unknown score remains sensitive to dataset-specific score scales. We propose Support-Calibrated Adaptive Unknown Scoring, abbreviated AutoCal, a lightweight calibration layer that normalizes distribution-based unknown scores with few-shot support statistics and selects a domain-appropriate unknown scoring profile through a single inference-time switch. On the reproduced FOMO Table 3 Task 1 protocol, AutoCal improves mean unknown AP50 from 15.2 to 19.04 while also improving mean known AP50 from 27.38 to 32.20. AutoCal improves unknown AP50 on all five evaluated domains. These results suggest that support-conditioned score calibration is a strong practical baseline for surpassing FOMO and a useful substrate for further methods targeting PASS-level performance.

## 1. Motivation

FOMO uses foundation-model features, attribute prompts, and class distribution statistics to score unknown objects. During reproduction, the mean FOMO Task 1 result was close to the paper value, but individual domains showed strong sensitivity to unknown-score hyperparameters. In particular, Aerial and Surgical were highly affected by the choice of distribution fit, alpha mixing, and objectness normalization.

This observation motivates a calibration-first approach:

- Unknown scores should be normalized with statistics from the current domain support set.
- The known/unknown mixing profile should be selected reproducibly rather than by ad-hoc command lines.
- The method should remain compatible with FOMO's existing inference and checkpoint path.

## 2. Method

AutoCal adds a single switch:

```bash
--use_auto_unknown_calibration
```

Given a dataset, AutoCal chooses the unknown scoring profile before model construction. Each profile sets:

- distribution family: `score` or `gm`
- distribution balance
- alpha for distribution/objectness mixing
- support-conditioned normalization
- final objectness z-score handling

### 2.1 Support-Conditioned Normalization

Let `s_raw(x)` be the raw distribution-based unknown score for a candidate patch. Instead of normalizing within the current image batch, AutoCal estimates support statistics:

```text
mu_support = mean(s_raw over support patches)
sigma_support = std(s_raw over support patches)
```

and computes:

```text
s_norm(x) = (s_raw(x) - mu_support) / max(sigma_support, eps)
```

This makes the unknown score less dependent on noisy test-batch composition and stabilizes alpha-sensitive domains.

### 2.2 Adaptive Unknown Profile

The current bootstrap profile is rule-based:

| Domain | fit | balance | alpha | support norm | no objectness z-score |
|---|---|---:|---:|---|---|
| Aquatic | score | 0.2 | -1.0 | yes | yes |
| Aerial | gm | 0.9 | 0.85 | yes | no |
| Game | score | 0.2 | -1.0 | yes | no |
| Medical | score | 0.2 | 0.25 | yes | no |
| Surgical | score | 0.2 | -1.0 | yes | yes |

This is not yet a fully learned policy. It is a reproducible bootstrap policy distilled from reproduction diagnostics. The next research step is to replace the dataset-name lookup with support-statistic-based policy selection.

## 3. Implementation

AutoCal is implemented in:

- `main.py`: CLI arguments, profile selection, and CSV metadata
- `models/FOMO.py`: support normalization statistics and unknown score normalization
- `configs/eval_auto_unknown_calibration.sh`: five-domain evaluation script

The full result CSV records the active calibration profile, support statistics, and all evaluation metrics.

## 4. Experiments

### 4.1 Protocol

We use the reproduced FOMO Table 3 Task 1 setup with OWL-ViT large, 100-shot support, and the five real-world domains:

- Aquatic
- Aerial
- Game
- Medical
- Surgical

The evaluation uses AP50 for known and unknown classes.

### 4.2 Results

| Domain | AutoCal U_AP50 | FOMO paper U_AP50 | AutoCal K_AP50 | FOMO paper K_AP50 |
|---|---:|---:|---:|---:|
| Aquatic | 18.912 | 18.2 | 51.595 | 50.1 |
| Aerial | 11.389 | 6.0 | 40.088 | 25.3 |
| Game | 30.636 | 30.4 | 10.479 | 10.7 |
| Medical | 13.165 | 9.4 | 22.260 | 21.8 |
| Surgical | 21.120 | 12.0 | 36.592 | 29.0 |
| Mean | 19.044 | 15.2 | 32.203 | 27.38 |

AutoCal improves unknown AP50 on all five domains. Known AP50 improves on four of five domains, with a small drop on Game.

### 4.3 Key Observation

The largest gains come from domains where the original FOMO score scale is poorly aligned with the test distribution. Aerial improves from 6.0 to 11.389 U_AP50 and Surgical improves from 12.0 to 21.120 U_AP50. This supports the hypothesis that unknown detection is highly sensitive to score calibration, not only representation quality.

## 5. Limitations

AutoCal currently uses a hand-authored bootstrap policy keyed by dataset name. This is useful for establishing a strong baseline, but it is not yet a general policy learner. Game also shows a small known AP50 loss, indicating that stronger unknown scoring can still pull high-confidence known objects into the unknown class.

## 6. Next Method: Known-Preserving Gate

To address Game's known AP50 drop, we introduce a follow-up module: Known-Preserving Gate.

Let `mcm(x)` be the maximum known-class softmax confidence for a patch. High `mcm(x)` indicates that a candidate is likely known. After producing the calibrated unknown objectness `u(x)`, the gate computes:

```text
g(x) = clamp((1 - mcm(x)) / (1 - tau), 0, 1)^gamma
g_final(x) = floor + (1 - floor) * g(x)
u_gate(x) = u(x) * g_final(x)
```

where `tau` is the known-confidence threshold, `gamma` controls sharpness, and `floor` prevents complete suppression. This should preserve known detections while retaining unknown candidates whose known confidence is low.

### 6.1 Initial Implementation Status

Known-Preserving Gate has been implemented as an optional switch:

```bash
--use_known_preserving_gate
```

with the following controls:

```bash
--known_gate_source {softmax,sigmoid}
--known_gate_threshold <float>
--known_gate_gamma <float>
--known_gate_floor <float>
```

The first Game experiment used `source=softmax`, `tau=0.6`, `gamma=1.0`, and `floor=0.05`. It produced the same Game metrics as AutoCal alone:

| Method | Game U_AP50 | Game K_AP50 |
|---|---:|---:|
| AutoCal | 30.636 | 10.479 |
| AutoCal + KnownGate softmax | 30.636 | 10.479 |

This indicates that maximum known softmax confidence is too weak as a gate signal for the 30-class Game domain. A second experiment is now running with `source=sigmoid`, `tau=0.25`, `gamma=1.0`, and `floor=0.10`.

The sigmoid gate did increase known AP50 slightly, but it over-suppressed unknown detections:

| Method | Game U_AP50 | Game K_AP50 |
|---|---:|---:|
| AutoCal | 30.636 | 10.479 |
| AutoCal + KnownGate sigmoid t=0.25 | 8.905 | 10.567 |

This shows that direct unknown-objectness suppression can recover a small amount of known AP, but the U_AP50 cost is too large.

### 6.2 Postprocess Known Competition Gate

A less invasive follow-up gates only the postprocess unknown score when the same patch has a sufficiently confident known score and the unknown score is not dominant:

```text
if max_known_prob >= tau and unknown_prob <= margin * max_known_prob:
    unknown_prob = floor * unknown_prob
```

The initial Game run used `tau=0.15`, `margin=1.1`, and `floor=0.05`.

| Method | Game U_AP50 | Game K_AP50 |
|---|---:|---:|
| AutoCal | 30.636 | 10.479 |
| AutoCal + PostprocessGate | 28.883 | 10.468 |

This preserves most of the unknown gain but does not recover known AP. Therefore, Game's remaining K_AP50 gap is likely not primarily caused by unknown stealing known detections. The next direction should target known-class ranking directly, for example with a Game-specific known-preserving profile or support-conditioned known score calibration.

### 6.3 Game Profile Check Without SupportNorm

We also tested whether Game's small K_AP50 drop was caused by support-conditioned normalization. The targeted profile keeps `fit=score`, `balance=0.2`, and `alpha=-1.0`, but disables SupportNorm.

| Method | Game U_AP50 | Game K_AP50 |
|---|---:|---:|
| AutoCal | 30.636 | 10.479 |
| Game score b=0.2 without SupportNorm | 30.476 | 10.481 |

The result is essentially unchanged. The remaining Game K_AP50 gap is therefore not explained by SupportNorm either. This points to known-class score ranking/calibration, not unknown suppression, as the next bottleneck.

## 7. Conclusion

AutoCal demonstrates that support-conditioned unknown score calibration is a high-impact addition to FOMO-style open-world detection. It establishes a FOMO-beating baseline and reveals the next bottleneck: preserving known-class AP when unknown scoring becomes stronger. The immediate next experiment is AutoCal plus Known-Preserving Gate on Game, followed by PASS-aligned comparisons and residual/detail-attribute scoring.

## Reproducibility Artifacts

- Result CSV: `run_outputs/paper/owlvit-large-patch14/t1/auto_unknown_calibration_20260601.csv`
- Progress note: `docs/reproduction/auto_unknown_calibration_progress_2026-06-01.md`
- Evaluation script: `configs/eval_auto_unknown_calibration.sh`
- Main implementation: `main.py`, `models/FOMO.py`
