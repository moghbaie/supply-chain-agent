# Step 3 summary: Coherent inventory simulation

## Objective and implementation
Built a daily simulator (`src/simulation/`, `configs/simulation.yaml`, seed 42) and ran it through `scripts/run_simulation.py`. Diagnostics are in `scripts/inspect_simulation.py`; output is in `run.log`.
Daily order of events: receive purchase orders, serve demand, then review and order. Unmet demand is lost, not backordered.

**Observed:** demand and prices come from M5. **Synthetic:** all inventory, suppliers, lead times, delays, purchase orders, receipts, costs and stockouts.

## Definitions
- Conservation: on_hand_end = on_hand_start + received - sales; demand = sales + unmet.
- Stockout day: unmet > 0.
- Label `stockout_next_14d`: any stockout day in t+1 to t+14; missing for the final 14 days.
- Policy: weekly Monday review. It sees only past simulated sales on in-stock open days and promised lead times.

## Data
- 500 series (419 items) sampled across category and state from 7,857 eligible. Funnel: 30,490 series, 21,389 launched by 2012-11-01, 7,857 with no 56+ day gap through 2014.
- Periods: burn-in 2013-01-01 to 03-31, train to 2014-12-31, validation 2015, test 2016-01-01 to 05-22.
- 3 scenarios: baseline, lean_policy, supplier_disruption. 619,000 daily rows each. Runtime 37 seconds.

## Results (open days, excluding burn-in)
| Scenario | Split | Fill % | Stockout-day % | 14-day label % |
|---|---|---:|---:|---:|
| baseline | train | 96.87 | 1.24 | 6.18 |
| baseline | validation | 96.26 | 1.26 | 6.09 |
| baseline | test | 96.72 | 1.29 | 6.31 |
| lean_policy | train | 93.52 | 2.81 | 14.44 |
| lean_policy | validation | 92.35 | 2.88 | 14.06 |
| lean_policy | test | 93.46 | 3.07 | 15.72 |
| supplier_disruption | train | 96.73 | 1.34 | 6.55 |
| supplier_disruption | validation | 95.87 | 1.41 | 6.64 |
| supplier_disruption | test | 95.57 | 1.86 | 8.64 |

- Baseline lead time: median 13 days, max 37; 15.95% of received orders delayed.
- For the 134 series on disrupted suppliers (SUP_02, SUP_07, SUP_11), during each window plus 30 days: fill about 96% in baseline and 87% under disruption.
- Conservation residual 0; minimum on hand 0.
- Plots: `stock_trajectories.png`, `lead_times.png`.

## Tests
49 of 49 passed; individual results are in `tests.log`. All passed on the first run, so no fixes were needed. Results are also reproducible across reruns and independent of series order.

## Limitations and unresolved risks
- **Selection bias:** the 56-day filter removes 63% of launched series. Selected series sell 2.00 units a day on average against 1.13 for all series, with 51.8% zero days against 68%. Results do not represent slow movers.
- **Sales used as demand:** M5 sales may already be limited by real stockouts, so true demand is understated.
- **Later gaps kept:** 103 of 500 series have a 56+ day zero gap after 2014, deliberately kept to avoid selecting on the future. Those zeros count as true demand.
- **Closure detection is partial:** only store-days with exactly zero sales count as closed, which covers 24 store-days. Other stores sold 0 to 11 units on Dec 25 and count as open.
- **Rules drive outcomes:** stockout patterns partly reflect this simulator's rules, so later model results on them do not prove real-world performance.
- **Placeholder values:** costs, supplier tiers, pack sizes and disruption settings are assumptions.
- **Supplier assignment is random:** supplier tier is not linked to item or category characteristics.
- **Expedite is not implemented as an action:** only its cost parameters exist, for Step 10.
- **Step 5 must purge labels:** 7,000 train rows per scenario have labels that look into validation, and validation has the same issue at its end. The last 7,000 test rows per scenario have no label.
- **Step 5 and 6 must use `purchase_orders_as_of`:** saved purchase orders contain outcomes that were unknown when placed.

## Acceptance criteria
- 20 inventory records, trajectories, lead-time distributions, stockout events and parameters printed: met.
- Conservation, nonnegative stock, chronology, reproducibility and edge cases tested: met.
- Train, validation and test periods defined: met.

## STOP: Awaiting user approval to proceed
