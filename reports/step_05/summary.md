# Step 5 summary: TabPFN stockout-risk model

## Objective and implementation
Predict the probability of a stockout within the next 14 days at each weekly review, using only what is known that day.
Code: `src/stockout/`, `scripts/build_stockout_forecasts.py`, `scripts/build_stockout_features.py`, `scripts/probe_tabpfn.py`, `scripts/run_stockout.py`.

**Synthetic data:** labels and inventory come from the Step 3 simulator; only demand and prices are from M5.

## Data
- 243,000 rows: 500 series, 162 Mondays, 3 scenarios, with 30 features.
- Train 135,000 (origins to 2014-12-15), validation 75,000 (2015), test 27,000 (2016-01-04 to 05-02). 6,000 rows whose label window crossed a boundary were purged.
- Positive rate on test: baseline 6.27%, supplier disruption 8.64%, lean policy 15.46%; 10.12% overall.
- Missing values: only `days_to_next_due` (39.88%), which is missing when no future delivery is due.
- Forecasts were regenerated with Timestral-28M from each scenario's simulated sales. Days ending with no stock are marked missing, because unmet demand is hidden from a planner.
- Hidden from features: unmet demand, future data, supplier tiers, scenario name and disruption windows.

## TabPFN settings (chosen on validation)
TabPFN v2 with 3,000 random training rows and 2 ensemble members, at about 12 ms per prediction. For test it was refit on a sample from train plus validation.

## Test results
| Model | Training rows | PR-AUC | ROC-AUC | Brier | ECE | Recall in top 5% |
|---|---:|---:|---:|---:|---:|---:|
| Constant rate | 210,000 | 0.1012 | 0.5000 | 0.0911 | 0.0328 | 2.49% |
| Days-of-cover rule | 210,000 | 0.4976 | 0.8730 | 0.0647 | 0.0081 | 33.59% |
| Small logistic | 210,000 | 0.5315 | 0.8843 | 0.0651 | 0.0178 | 34.17% |
| **TabPFN v2** | **3,000** | **0.6491** | **0.9103** | **0.0560** | **0.0185** | **38.20%** |
| LightGBM | 210,000 | 0.6847 | 0.9218 | 0.0524 | 0.0077 | 40.25% |

TabPFN minus rival, series bootstrap with 95% intervals:
- Cover rule: PR-AUC +0.151 (0.136 to 0.166); Brier -0.0087 (-0.0103 to -0.0072).
- Small logistic: PR-AUC +0.118 (0.099 to 0.134); Brier -0.0092 (-0.0108 to -0.0077).
- LightGBM: PR-AUC -0.036 (-0.046 to -0.026); Brier +0.0035 (0.0026 to 0.0046).

LightGBM is ahead of TabPFN in all three scenarios. Plots and tables: `calibration.png`, `reliability_*.csv`, `metrics_by_scenario.csv`, `predictions_preview.csv`.

## Tests
28 of 28 passed; individual results are in `tests.log`. All passed on the first run. `test_no_infinite_feature_values` was then corrected because its original form could not fail, and the suite was rerun.

## Limitations and unresolved risks
- **TabPFN is not the best model here.** It beats the simple baselines but loses to LightGBM on all data. On the same small samples TabPFN did better (validation sample, PR-AUC 0.613 at 1,000 rows against 0.542), so its edge is with limited data. **Replacing it needs user approval.**
- **TabPFN underpredicts slightly:** mean predicted 8.27% against 10.12% observed, with the largest gaps in the middle of the range (5.3% predicted, 9.6% observed in bin 8). No recalibration was applied.
- **Training-sample sensitivity:** on 5,000 validation rows, PR-AUC was 0.6153, 0.6195 and 0.6306 across three samples. The chosen seed scored highest, so validation figures may be slightly flattering. It was not chosen for that reason, since the seed was fixed in advance.
- **TabPFN v3.5 was not tested.** The Prior Labs license check returned not accepted for the token in use, despite the user reporting acceptance.
- **Few training rows, small settings:** 3,000 rows and 2 members were chosen for speed on a 4 GB GPU. More could narrow the gap.
- **Runtime:** the logged 32,458.8 seconds for TabPFN on test is not a real speed; it likely includes the laptop sleeping overnight, though this is unconfirmed. Validation took 986.6 seconds for 75,000 rows.
- **The model learns simulator rules:** these scores do not show real-world performance.
- **Timestral license:** the $20M monthly revenue limit for a company or employer remains unconfirmed. The project is research only, but that does not settle this condition.
- **Severity is not modeled:** the label treats a 1-unit miss the same as a large one.

## Acceptance criteria
- Feature schema, 20 rows, class balance, 20 predictions, PR-AUC, Brier and calibration plots: met.
- Label, feature availability, chronological split, missing value and class imbalance tests: met.
- Calibrated, reproducible predictions beating a simple baseline: met, with the slight underprediction noted.

## STOP: Awaiting user approval to proceed
