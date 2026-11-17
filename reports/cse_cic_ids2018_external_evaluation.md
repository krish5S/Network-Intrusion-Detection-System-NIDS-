# External XGBoost evaluation: CSE-CIC-IDS2018

- **Evaluation date:** 2026-11-17
- **Dataset:** Thursday, 1 March 2018 CICFlowMeter CSV from the official CSE-CIC-IDS2018 S3 bucket
- **Saved model:** `models/xgb_model.joblib` with `models/scaler.joblib`
- **Decision threshold:** 0.50

## Dataset and method

CSE-CIC-IDS2018 is a later, related network intrusion dataset. The official dataset description says its flow statistics were extracted with CICFlowMeter-V3 and includes infiltration among its attack scenarios. This evaluation used one official day file, which contains benign and infiltration traffic; it is a focused external check, not a broad evaluation of every 2018 attack family. [Dataset description](https://www.unb.ca/cic/datasets/ids-2018.html) · [Official AWS data listing](https://registry.opendata.aws/cse-cic-ids2018/)

The source CSV had 331,125 rows. The workflow discarded 2,944 invalid rows (including repeated header records and rows with invalid numeric values), then removed 40,154 exact duplicate feature-vector/label rows. The final scored set had **288,027 flows**: **205,496 benign** and **82,531 attacks**. The source attack label was `Infilteration`; all non-benign, non-header labels were mapped to attack.

All 77 required model inputs were present. CICFlowMeter-V3 abbreviations were mapped to the project's feature names (for example, `Dst Port` → `Destination Port`, `Tot Fwd Pkts` → `Total Fwd Packets`, and `Flow Byts/s` → `Flow Bytes/s`). `Protocol` and `Timestamp` were excluded because they are not model inputs. No missing feature values were fabricated. The saved scaler and classifier were used unchanged; neither was fit on the evaluation data.

The evaluated artifact is a **100-tree** XGBoost model (`binary:logistic`, depth 6, learning rate 0.1), the artifact currently loaded by the dashboard. The **99.92%** result in `notebooks/models.ipynb` is from a separate 200-tree notebook benchmark, so it is not a like-for-like score comparison.

## Results at threshold 0.50

| Metric | Result |
|---|---:|
| Accuracy | 70.99% |
| Attack precision | 41.23% |
| Attack recall | 2.89% |
| Attack F1 | 5.41% |
| Macro F1 | 0.4414 |
| False-positive rate | 1.66% |
| ROC-AUC | 0.5807 |
| Average precision (PR-AUC) | 0.3897 |
| Attack-probability Brier score | 0.2663 |

Confusion matrix (actual rows, predicted columns):

|  | Predicted benign | Predicted attack |
|---|---:|---:|
| Actual benign | 202,093 | 3,403 |
| Actual attack | 80,144 | 2,387 |

The model detected only 2,387 of 82,531 attacks at the default threshold. The 70.99% accuracy therefore hides poor attack detection. The ranking metrics are also weak on this evaluation set. This is evidence that the saved model does **not** transfer reliably to this CSE-CIC-IDS2018 infiltration slice as-is.

## Threshold sensitivity

Lowering the threshold changes the operating point, but does not fix transfer performance. At 0.05, attack recall rises to 25.65% while the false-positive rate rises to 10.09%. This is diagnostic only; the threshold was not tuned or selected on this external evaluation set.

| Threshold | Attack precision | Attack recall | False-positive rate |
|---:|---:|---:|---:|
| 0.05 | 50.51% | 25.65% | 10.09% |
| 0.10 | 58.69% | 18.69% | 5.28% |
| 0.25 | 43.93% | 4.96% | 2.54% |
| 0.50 | 41.23% | 2.89% | 1.66% |
| 0.75 | 15.12% | 0.20% | 0.46% |
| 0.95 | 17.31% | 0.07% | 0.13% |

The attack-probability Brier score is recorded as a probability-quality diagnostic; the scores should not be presented as calibrated confidence. Ticket 002 exposes both XGBoost class probability outputs while labeling them as model estimates.

## Reproduction

From the project root, run:

```bash
./.venv/bin/python src/evaluate_external.py
```

The script expects the selected CSV at `data/external/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` and writes machine-readable results to `reports/cse_cic_ids2018_external_evaluation.json`. The local CSV SHA-256 is `b0534c5d7d8b41e03df71c6966c995d116a8ed28e61f377c8b14cdf5d28f4edf`.
