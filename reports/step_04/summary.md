# Step 4 summary: Demand forecasting

## Objective and implementation
Verified Timestral's interface and benchmarked it against two baselines on unseen periods.
Code: `src/forecasting/`, `configs/forecasting.yaml`, `scripts/run_forecasting.py`, `scripts/inspect_forecasts.py`.
Timestral 1.0.0 was installed from github.com/mistralai/timestral at commit a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff.

**Observed:** all figures use real M5 sales for the 500 Step 3 series, not simulated sales.

## Method
Rolling-origin evaluation from Mondays, forecasting 28 days, with sales known through each date. Validation has 48 dates (targets Jan 6 to Dec 28, 2015) and test has 16 dates (targets Jan 5 to May 16, 2016). Closed-store days and days before first sale are excluded.
Baselines: seasonal naive, with empirical quantiles from the last 56 days, and LightGBM, trained on targets through 2014 for validation (1,399,097 rows) and through 2015 for test (2,125,947 rows).

## Variant selection (12 validation dates, rule fixed in advance)
| Variant | Pinball per unit | Seconds per date |
|---|---:|---:|
| 5M, none | 0.6225 | 0.7 |
| 5M, calendar | 0.6250 | 9.7 |
| 28M, none | 0.6185 | 3.3 |
| 28M, calendar | 0.6185 | 38.1 |
| 123M, none | 0.6142 | 11.9 |
| 209M, none | 0.6136 | 21.3 |

Chosen: Timestral-28M with no covariates, the smallest within 1% of the best.
Deviation: calendar variants for 123M and 209M were not run, after calendar inputs ran 11-13x slower with no gain at 5M and 28M. The first run was stopped for this reason and its 5 finished variants were kept.

## Test results (16 dates)
| Model | WAPE, median % | WAPE, mean % | Bias, mean % | Bias, median % | Pinball | 80% coverage % | 14-day total WAPE % | 28-day total WAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seasonal naive | 64.70 | 80.41 | -0.38 | -24.66 | 0.627 | 89.42 | 33.84 | 32.72 |
| LightGBM | 61.20 | 66.24 | -1.55 | -23.80 | 0.592 | 87.86 | 28.03 | 25.76 |
| Timestral-28M | 60.11 | 64.88 | -0.17 | -22.33 | 0.580 | 84.42 | 25.55 | 23.35 |

Pinball ratio, Timestral to rival (series bootstrap, 95% interval):
- Seasonal naive: 0.924 (0.899 to 0.948); Timestral better on 55.8% of series.
- LightGBM: 0.979 (0.970 to 0.987); Timestral better on 47.8% of series.

By volume quartile, ratio to LightGBM: slowest 1.003, second 0.998, third 0.995, fastest 0.968. The fastest quartile holds 74.0% of units.
Peak VRAM 983 MB. Timestral-28M took 61.7 seconds for all 16 test dates.

## Tests
25 of 25 passed after one fix; individual results are in `tests.log`.
`test_gbm_training_ignores_data_after_cutoff` first failed because the test forecast from a date after its own cutoff, so its inputs differed by construction. It now forecasts from before the cutoff. Training deliberately past the cutoff makes the outputs differ, as checked separately in this step.

## Limitations and unresolved risks
- **Modest win, concentrated in fast sellers:** on slow sellers Timestral is roughly level with LightGBM.
- **Daily forecasts are noisy for every model:** demand is zero on 50% of test days, so totals over 14 or 28 days are the more decision-relevant measure.
- **Median forecasts run about 22% low:** this is typical of intermittent demand. Use the mean for expected demand and upper quantiles for risk.
- **Intervals:** 84.4% coverage against a nominal 80%. The upper tail is on target (10.0% above q90); the lower side is 5.6% below q10, helped by the floor at zero.
- **LightGBM setup:** one fixed configuration, not tuned, and not retrained weekly, which may understate it.
- **Series selection:** the 500 series lean toward faster sellers (Step 3), so results do not generalize to slow movers.
- **Test period:** January to mid-May only, with no year-end holiday season. Comparing test with validation scores is not fair.
- **Spikes are missed:** for example HOBBIES_1_398_CA_3 sold 59 units on one day against a mean forecast near 3.
- **Step 5 must regenerate forecasts:** these use real M5 sales, while Step 5 needs as-of-date forecasts from each scenario's simulated sales, with stockout days marked missing.
- **License:** the Timestral weights' Modified MIT license withholds rights if your company or employer has monthly revenue above $20M, including for derivatives. It does not exempt research. Timestral's code is Apache-2.0.
- **Benchmarks reported by Mistral were not reproduced here.**

## Acceptance criteria
- Actual vs predicted for 10 SKUs, 14-28-day forecasts, intervals, WAPE and bias: met; see `ten_skus.csv`, `forecast_vs_actual.png`, `inspection.log`.
- Seasonal naive and boosted baselines: met.
- Leakage, horizon, interval, stability and baseline tests: met.
- Documented performance on unseen periods: met.

## STOP: Awaiting user approval to proceed
