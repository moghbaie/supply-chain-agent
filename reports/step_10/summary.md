# Step 10: End-to-end replay

**Status: complete, waiting for approval.** All inventory, supplier and cost figures are simulated; only demand and prices come from M5. Nothing left this machine in this step.

## Main result

- **The agent gives the best service but costs far more.** It cuts unmet demand by 71% to 75% against the reorder point, and total cost rises 75% to 167%.
- **Forecast-driven is the cheapest run in all three scenarios.** It clearly beats the reorder point on both cost and unmet demand.
- **Expediting drives the extra cost.** It makes up 47% to 68% of the agent's cost. Against forecast-driven, the agent pays $8 to $11 for each lost unit avoided, after counting the margin it saves.
- **The agent is not clearly better than a simple policy rule.** Cost is not clearly different in any scenario, and unmet demand is clearly lower only under supplier disruption. It raises about 1.6 to 1.8 times as many cases for people to act on.
- **The LLM still adds almost nothing.** It picked UltraBERT's category on 99 of 100 sampled cases, and both changes to the final result came from errors, not judgment.
- **The approval boundary held.** No order was placed without a simulated person's approval. In tests on 30 series with every action declined, the replay matched the plain routine exactly.

These results do not show the agent is worse in general. They depend on the simulator and on costs that count only the 30% margin for a lost sale.

## What was built

| Part | File | Notes |
|---|---|---|
| Replay engine | `src/evaluation/replay.py` | Daily closed loop: each run's orders change its own later state |
| Model pipeline on replayed states | `src/evaluation/scoring.py` | Timestral-28M, TabPFN v2, LightGBM and fine-tuned UltraBERT, refit or loaded as in Steps 4 to 8 |
| Decisions and simulated person | `src/evaluation/decisions.py` | Code checks, conflict rule and approval |
| Runs and scoring | `scripts/run_replay.py`, `score_replay.py`, `build_replay_sample.py`, `score_llm_sample.py` | Finished runs are saved, and an interrupted replay resumes |
| Conflict rule in the live agent | `src/agent/review.py`, `src/agent/runner.py`, `scripts/run_agent.py` | Off by default, so the Step 8 baseline is unchanged |

### Replay design

- **Period and data:** 500 series, 2016-01-01 to 2016-05-22 (143 days, 20 Monday reviews, 71,500 daily rows per run), three scenarios. Every run starts from the saved simulation state on the day before.
- **Same constraints:** demand, prices, closed days and starting stock are the same for all runs in a scenario. An order placed for the same series on the same day meets the same delay draw.
- **Costs (`configs/simulation.yaml`):** holding 25% a year, lost margin 30% of price, shipping 5% of price on receipt. An expedite adds the same shipping again plus a $25 fee.
- **Total cost excludes** the purchase price of units and reviewers' time.

| Run | Rule |
|---|---|
| `reorder_point` | The simulator's routine, based on recent sales |
| `forecast_driven` | Same formula and safety factor, using the Timestral forecast instead of past sales |
| `policy_layer` | Reorder point, plus the Step 7 policy rule's recommendations |
| `agent` | Reorder point, plus UltraBERT and code checks, with no LLM |
| `agent_conflict` | Same, and a gap of 30 points or more between stockout models goes to human review |
| `agent_conflict_decline_review` | Same, and the person declines human-review cases |
| `agent_conflict_weak_expedite` | Same, and an expedite cuts 25% of the remaining wait instead of 50% |

**Assumptions I set before running, none of them measured:**
- An expedite halves the remaining wait, with one expedite per order, on the soonest order not yet expedited.
- The simulated person approves every recommendation in the main runs. A human-review case becomes an expedite if an order is open, otherwise an order.
- An agent order fills up to the order-up-to level, at least one pack.

## Data checks before the runs

| Check | Result |
|---|---|
| Replay with saved delays against the Step 3 simulation (`regression.log`) | 0 of 71,500 rows differ on 9 columns, all 3 scenarios |
| Replay's own delay draws against saved totals, unmet units | Baseline 4,566 vs 4,564; lean 9,134 vs 9,093; disruption 6,035 vs 6,157 |
| Replayed features and models against saved test cases, 40 days, 6 reviews, 3,000 cases per scenario (`pipeline_regression.log`) | Same UltraBERT category on 9,000 of 9,000; features exact; stockout risk gap up to 0.0005; UltraBERT probability gap up to 0.0146 |
| Starting state | Baseline 21,650 units and 246 open orders; lean 14,976 and 349; disruption 21,738 and 259 |

## Results

Full tables: `replay_metrics.csv`, `replay_differences.csv`, `replay_scores.log`.

### Baseline

| Run | Stockout days % | Unmet units | Holding | Lost margin | Shipping | Expedite | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reorder point | 1.33 | 4,566 | 5,425 | 3,997 | 18,253 | 0 | 27,675 |
| **Forecast-driven** | 1.02 | 3,403 | 5,470 | 2,791 | 18,427 | 0 | **26,687** |
| Policy rule | 0.36 | 1,183 | 5,650 | 1,058 | 18,862 | 22,998 | 48,568 |
| Agent, no LLM | 0.30 | 1,151 | 5,835 | 1,084 | 18,929 | 22,576 | 48,424 |
| Agent with conflict rule | 0.30 | 1,151 | 5,835 | 1,084 | 18,929 | 22,576 | 48,424 |
| Conflict rule, human review declined | 0.48 | 2,304 | 5,773 | 2,217 | 18,619 | 19,892 | 46,499 |
| Conflict rule, weaker expedites | 0.65 | 2,211 | 5,696 | 2,043 | 18,704 | 24,505 | 50,948 |

### Lean policy

| Run | Stockout days % | Unmet units | Holding | Lost margin | Shipping | Expedite | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reorder point | 3.04 | 9,134 | 3,664 | 8,153 | 17,451 | 0 | 29,268 |
| **Forecast-driven** | 2.55 | 7,025 | 3,740 | 6,037 | 17,921 | 0 | **27,698** |
| Policy rule | 0.89 | 2,610 | 4,030 | 2,527 | 18,488 | 54,204 | 79,249 |
| Agent, no LLM | 0.73 | 2,376 | 4,241 | 2,322 | 18,741 | 52,764 | 78,069 |
| Agent with conflict rule | 0.73 | 2,376 | 4,241 | 2,322 | 18,741 | 52,764 | 78,069 |
| Conflict rule, human review declined | 1.09 | 4,334 | 4,172 | 3,890 | 18,456 | 48,120 | 74,638 |
| Conflict rule, weaker expedites | 1.50 | 4,826 | 4,049 | 4,420 | 18,253 | 56,270 | 82,992 |

### Supplier disruption

| Run | Stockout days % | Unmet units | Holding | Lost margin | Shipping | Expedite | Total cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reorder point | 1.86 | 6,035 | 5,336 | 5,153 | 18,062 | 0 | 28,551 |
| **Forecast-driven** | 1.52 | 4,713 | 5,388 | 3,830 | 18,158 | 0 | **27,376** |
| Policy rule | 0.60 | 2,096 | 5,602 | 1,768 | 18,874 | 26,681 | 52,925 |
| Agent, no LLM | 0.51 | 1,749 | 5,782 | 1,341 | 18,956 | 25,697 | 51,776 |
| Agent with conflict rule | 0.51 | 1,749 | 5,782 | 1,341 | 18,956 | 25,697 | 51,776 |
| Conflict rule, human review declined | 0.69 | 2,732 | 5,734 | 2,698 | 18,603 | 23,259 | 50,293 |
| Conflict rule, weaker expedites | 0.98 | 3,365 | 5,624 | 2,828 | 18,584 | 27,757 | 54,793 |

### Paired differences, series bootstrap, 1,000 rounds [2.5%, 97.5%]

Negative favors the first-named run. Bold intervals exclude zero.

| Comparison | Measure | Baseline | Lean policy | Supplier disruption |
|---|---|---|---|---|
| Forecast-driven minus reorder point | Cost | **-1,913, -251** | **-3,039, -554** | **-1,975, -533** |
| | Unmet | **-2,039, -426** | **-3,432, -1,015** | **-2,377, -481** |
| Agent with conflict rule minus forecast-driven | Cost | **19,250, 24,421** | **46,532, 54,551** | **21,736, 27,324** |
| | Unmet | **-3,233, -1,387** | **-5,957, -3,612** | **-3,731, -2,283** |
| Agent with conflict rule minus policy rule | Cost | -1,629, 1,215 | -3,090, 778 | -2,399, 182 |
| | Unmet | -242, 238 | -476, 63 | **-688, -6** |

### What the cost result depends on

Actions never depend on these costs, so this needed no rerun (`cost_sensitivity.csv`). Agent with conflict rule against forecast-driven:

| | Baseline | Lean policy | Supplier disruption |
|---|---:|---:|---:|
| Extra total cost | 21,736 | 50,371 | 24,400 |
| Of which $25 fees | 17,350 | 45,125 | 19,850 |
| Unmet units avoided | 2,252 | 4,649 | 2,964 |
| Extra cost per unit avoided | 9.65 | 10.83 | 8.23 |
| Same with no fee at all | 1.95 | 1.13 | 1.54 |
| Margin counted per lost unit | 0.82 | 0.86 | 0.81 |

The extra cost is already net of the margin saved. So the agent pays off only if a lost unit costs about $8 to $11 more than its counted margin, or $1.13 to $1.95 more if expedites had no fee. The mean price is about $4.30. Whether a lost sale costs that much, for example through lost customers, is a business question this project cannot answer.

### Workload and errors

| Agent with conflict rule | Baseline | Lean policy | Supplier disruption |
|---|---:|---:|---:|
| Cases reviewed | 10,000 | 10,000 | 10,000 |
| Cases needing a person's action | 1,521 | 3,403 | 1,619 |
| Same for the policy rule | 842 | 2,171 | 970 |
| Human-review cases | 101 | 195 | 103 |
| Moved to human review by the conflict rule | 62 | 119 | 66 |
| Actions with both stockout risks under 10% | 673 (44%) | 1,140 (34%) | 667 (41%) |
| Approved expedites that could gain no time | 80 | 121 | 95 |
| Step 7 overrides fired | 0 | 0 | 0 |
| Missing stockout scores | 0 | 0 | 0 |

No replay run failed.

**The conflict rule changed no outcome when everything was approved.** All three scenarios give the same results with and without it, by construction: the overrides already make expedite cases have an open order and reorder cases have none, so human review leads to the same action. No monitor case had a gap of 30 points. The rule only matters when a person can decline, as in the decline run.

**Declining human review has a large effect for a real reason.** Those cases are about 1% to 2% of all cases, but the items are high volume, with a mean of about 102 to 114 forecast units over 14 days against about 27 overall. Declining them nearly doubles unmet demand on baseline and lean policy and adds 56% on disruption, and nearly all the extra unmet units are in series sent to human review. It also lowers total cost, since expedites are the main cost.

### Example recommendations (baseline, first review)

| Item | On hand | On order | Days of cover | Forecast, 14 days | Risk, TabPFN / LightGBM | Final | Action |
|---|---:|---:|---:|---:|---|---|---|
| FOODS_1_002_CA_4 | 17 | 0 | 107 | 2.2 | 0.1% / 0.1% | monitor | none |
| FOODS_1_018_TX_3 | 38 | 222 | 4.3 | 123.6 | 11.4% / 42.5% | human review (models conflict) | expedite |
| FOODS_1_018_WI_1 | 44 | 114 | 5.8 | 105.4 | 47.3% / 36.9% | expedite | expedite |
| FOODS_1_121_CA_3 | 30 | 0 | 44 | 9.5 | 0.4% / 2.6% | reorder | order |

The last row shows the weak-evidence problem: a reorder at 44 days of cover, with both models under 3%. More rows are in `example_recommendations.csv`.

## LLM sample

SmolLM3-3B ran on 100 replayed cases from the `agent_conflict` runs, 25 per final category, seed 42, with the conflict rule on. **You reduced this from the planned 300 to save time.** The sample reviews states after the fact: the LLM did not drive the replay, so the costs above are code-only. Results are in `llm_sample_scores.log` and `llm_sample_results.csv`.

| Measure | Result |
|---|---|
| Chose UltraBERT's category | 99 of 100; the other case gave no valid answer |
| Final result same as code only | 98 of 100 |
| Orders placed | 0 |
| Failed runs | 1, sent to a person |
| Malformed calls, invalid submissions | 29, 4 |
| `search_policy` calls | 0 |
| Seconds per case | Median 73, mean 80, 95th percentile 122, maximum 184 |
| Tokens per case | Mean 8,156 in, 261 out |
| Peak GPU memory | 2.23 GB |
| Total time | 7,958 s (2 h 13 min) |

The two changed cases both moved monitor to human review because of errors:
- **replay-0024:** the LLM read a 0.4% risk as "40%", the same kind of mistake seen in Step 8, and the number check caught it.
- **replay-0067:** three submissions were wrapped incorrectly, so the run ended with no valid answer.

Monitor is about 78% of real cases. At 2 of 25 monitor cases, the LLM would add roughly 1,900 human reviews per 30,000 cases, but that estimate rests on 2 cases and is very rough.

**Explanation audit**, using the Step 8 rules, which give a lower bound:

| Rule broken | Cases |
|---|---:|
| Any rule | 41 |
| Any factual rule | 26 |
| Cites no numbers | 22 |
| Says high risk, both models under 20% | 9 |
| Says no open orders, one exists | 8 |
| Says it conflicts with the suggestion it followed | 5 |
| Risk number from neither model | 4 |
| Says delayed, nothing overdue | 1 |

1 of 23 final monitor cases has a factual error that no person would review. I read the first two explanations per category printed in `llm_sample_scores.log`, not all 100, so this count is rule-based only.

## Tests

- `tests/integration/test_replay.py`: 112 passed; per-test results in `tests_replay.log`. No test needs the GPU or a hosted API.
- Full suite: 469 passed, 0 failed (`tests_full.log`). That is the 357 from Step 8 plus these 112.
- The Step 8 agent tests still pass after the conflict rule was added (79 of 79).

| Type | Covered |
|---|---|
| Integration | All 21 runs complete and conserve stock; every review decided every series; saved metrics match the daily rows |
| Identical constraints | Same demand, prices, closed days, starting stock and delay draws across runs |
| Policy compliance | Decline-all equals the routine exactly; monitor changes nothing; series with no approved action match the routine; conflict rule; LLM traces place no orders |
| Recovery | Finished runs are not repeated; the file that marks a run finished is written last; a model failure stops the run; an order with no arrival is refused; an agent with no answer goes to a person |
| Regression | Replay reproduces the simulation; saved routine runs repeat exactly; decisions repeat; the live agent's default keeps Step 8's result |
| Stress | Zero demand, demand spike, every supplier delayed, store closed throughout, missing scores |
| No leakage | Changing future demand leaves earlier decisions unchanged; hidden delays do not reach cases before arrival; the forecaster sees only days already played |

**Do the tests catch real faults?** All 112 passed on first run, so I broke the code on purpose in 6 ways: ignoring decline-all, disabling the conflict rule, showing the forecaster all days, leaking tomorrow's demand, removing the expedite fee and breaking stock conservation. The tests caught all 6, and the code was restored.

**Failures fixed during this step:**
- The first replay regression gave orders never received a delay of 0, which caused small differences.
- The first model regression fit LightGBM on rows in a different order from Step 5, moving risk by up to 0.29. Keeping Step 5's row order fixed it.
- Expedited older orders appeared still open after arrival, and one variable was undefined. Both were found on re-reading the code before the full run.
- The LLM scoring script crashed on a column named `flags`, which clashes with a pandas attribute.

## Departures from the plan

| Departure | Decided by |
|---|---|
| Step 9 skipped, so the agent is untuned | User |
| LightGBM replaces Kumo, UltraBERT replaces Jev, SmolLM3-3B replaces Qwen3-4B | User |
| LLM on 100 sampled cases, not in the replay loop | User, on my recommendation |
| Conflict rule added after Step 8; both versions reported | User, on my recommendation |
| Olist not replayed, since it has no inventory | Me |
| No separate demo was built, though the plan's title mentions one; `test_traces.txt` from Step 8 and the examples here are the closest | Me |
| No plots; tables only | Me |

## Limitations and unresolved risks

- **One simulator.** Supplier delays, inventory and costs are assumptions, so none of this measures real savings.
- **Key assumptions drive the result.** The time saved by an expedite, the $25 fee and the value of a lost sale were not measured. Weaker expedites nearly double the agent's unmet demand on baseline.
- **Lost sales count only margin.** No customer loss or penalty is included, which favors the cheaper runs.
- **Excluded costs.** Unit purchase price and review labour are left out. Agent runs generally end with more stock on hand and less on order, so totals are not directly comparable to the last dollar. Baseline ending stock is $61,704 for the agent against $57,252 for forecast-driven.
- **The simulated person is not a person.** A real reviewer might decline the weak cases, which could lower both cost and benefit. That was not tested beyond the two sensitivity runs.
- **The agent can only add to the routine.** It cannot cancel or shrink a routine order.
- **The two main approaches are not on equal terms.** The agent sits on the reorder-point routine, not on forecast-driven levels. Combining them was not tested and might be better than either.
- **Critical recall stays low**, as already accepted (0.41 in Step 7).
- **LLM sample limits.** It covers 100 cases, sampled evenly across categories rather than at real rates, so rare events are poorly measured. Its explanation errors stay unreviewed in monitor cases.
- **Full model runs were not repeated end to end.** Repeatability is shown by regression tests and matching runs, not by rerunning all 21. The `agent` and `agent_conflict` runs did give identical results independently.
- **Fresh clones cannot run past Step 6 without the saved Step 7 files,** as noted in the README.
- **Timestral's $20M revenue license limit** remains unconfirmed, and was assumed not to apply to this research project.

## Acceptance criteria

| Criterion | Assessment |
|---|---|
| Replay on unseen data against both baselines under identical constraints | Met and tested |
| Stockout rate, unmet demand, holding, expedite and total cost, workload, errors, examples | Met |
| Integration, recovery, policy compliance, regression and stress tests | Met, 112 of 112, and 469 of 469 in the full suite |
| Reproducible end-to-end results | Mostly met: seeds and code are fixed and regressions pass, but all 21 runs were not repeated, and a fresh clone needs the saved Step 7 files |
| Explicit limitations | Met |
| Human approval for purchases | Met in simulation: nothing is ordered without a simulated approval, and no real orders exist |

## STOP: Awaiting user approval to proceed
