# Supply Chain Reasoning Agent

A research project that tests whether an LLM agent coordinating specialized models makes better inventory decisions than simple replenishment rules. For each item at each weekly review, the agent recommends one of four actions: monitor, reorder, expedite or send to human review.

**The agent never places an order.** Every action other than monitor needs a person's approval, and code checks this after the model runs.

> **Research only.** The inventory data is simulated and the project has no connection to a live purchasing system. Olist is licensed for non-commercial use only, and the M5 and model licenses have not been reviewed for commercial use.

## Status

| Step | What it covers | Status |
|---|---|---|
| 1 | Environment and tool checks | Approved |
| 2 | Download and inspect M5 and Olist | Approved |
| 3 | Inventory simulator | Approved |
| 4 | Demand forecasting | Approved |
| 5 | Stockout risk | Approved |
| 6 | Delay risk | Approved |
| 7 | Escalation classification | Approved |
| 8 | Untuned agent baseline | Approved |
| 9 | Fine-tune the reasoning LLM | Skipped by decision |
| 10 | End-to-end replay against baselines | Finished, awaiting approval |

Each finished step has its method, data, results, tests and limitations in `reports/step_XX/summary.md`. The original plan is in [`sca_plan.md`](sca_plan.md).

## How it works

```text
M5 demand and prices (observed)
        |
        v
Inventory simulator: stock, suppliers, purchase orders, delays (synthetic)
        |
        v
Weekly review of each item
  - Timestral-28M              28-day demand forecast
  - TabPFN v2 and LightGBM     chance of a stockout in the next 14 days
  - LightGBM                   chance a supplier delivers late
  - Fine-tuned UltraBERT       suggested category
        |
        v
SmolLM3-3B agent               calls typed tools, writes an explanation
        |
        v
Code checks                    overrides, unsupported claims, approval rules
        |
        v
Recommendation for a person    nothing is ordered automatically
```

The models and the agent run locally. Sending data to the two hosted models used earlier in the project, Jev and Kumo, is now turned off in code.

## Data

| Source | Used for | Type |
|---|---|---|
| [M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) | Demand and prices for 500 sampled item-store series | Observed |
| [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | Separate benchmark of late customer deliveries | Observed |
| Simulator | Inventory, suppliers, lead times, purchase orders, delays, costs and stockouts | Synthetic |

M5 and Olist describe unrelated companies and are never joined. Neither contains inventory or inbound purchase orders, so **every stockout, escalation and replay result reflects this simulator's assumptions, not a real business.**

The simulator has three scenarios: `baseline`, `lean_policy` and `supplier_disruption`. Splits are chronological: train through 2014, validation in 2015, and test from 2016-01-01 to 2016-05-22.

## Results so far

All figures are on unseen test periods. Stockout and escalation results use simulator data.

| Task | Model used | Result | Comparison |
|---|---|---|---|
| Stockout risk (PR-AUC) | TabPFN v2 | 0.649 | LightGBM 0.685, days-of-cover rule 0.498 |
| Supplier delay (PR-AUC) | LightGBM | 0.336 | Supplier's past late rate 0.312, Kumo 0.325 |
| Escalation (macro-F1) | Fine-tuned UltraBERT | 0.492 | Policy rule 0.491, Jev 0.428 |
| Escalation (critical recall) | Fine-tuned UltraBERT | 0.413 | Policy rule 0.384, Jev 0.236 |

Forecast results are in [`reports/step_04/summary.md`](reports/step_04/summary.md).

The main findings so far:

- **UltraBERT does not clearly beat the policy rule.** Its macro-F1 and critical recall are not clearly different. It acts on more real stockouts, but its false alarms are about double (20.5% against 10.8%).
- **It still misses most severe cases.** Critical recall of 0.41 was accepted as a known limit.
- **The LLM added no independent decision.** In Step 8, SmolLM3-3B chose the same category as UltraBERT on all 100 test cases.
- **Its explanations are unreliable.** 35 of the 100 had a detectable factual error, including 5 of the 24 monitor cases, which no person would review.
- **The LLM is slow.** It takes about 80 seconds per case on the 4 GB GPU, so Step 10 runs it on a sample only.

### End-to-end replay (Step 10)

Simulated test period, 500 series. Total cost excludes the purchase price of units and review time. The agent runs here use code and models only, with the LLM reviewed on a separate sample.

| Run | Baseline: unmet units | Baseline: total cost | Lean policy: unmet units | Lean policy: total cost | Disruption: unmet units | Disruption: total cost |
|---|---:|---:|---:|---:|---:|---:|
| Reorder point | 4,566 | $27,675 | 9,134 | $29,268 | 6,035 | $28,551 |
| Forecast-driven | 3,403 | **$26,687** | 7,025 | **$27,698** | 4,713 | **$27,376** |
| Policy rule | 1,183 | $48,568 | 2,610 | $79,249 | 2,096 | $52,925 |
| Agent | 1,151 | $48,424 | 2,376 | $78,069 | 1,749 | $51,776 |

- **The agent has the best service, and forecast-driven is the cheapest.** Expediting makes up 47% to 68% of the agent's cost.
- **The agent pays off only if lost sales cost much more than assumed:** about $8 to $11 above the counted margin per unit, or $1.13 to $1.95 with no expedite fee.
- **The agent is not clearly better than the simple policy rule,** and it raises about 1.6 to 1.8 times as many cases for people to act on.
- **The LLM chose UltraBERT's category on 99 of 100 sampled cases.** Both changes to a final result came from errors.

These results depend on assumptions, including how much time an expedite saves. See [`reports/step_10/summary.md`](reports/step_10/summary.md).

## Setup

Developed on Windows 11 with Python 3.12.10, PyTorch 2.11 (CUDA 12.8) and a 4 GB GPU. The Step 8 agent peaked at 2.20 GB of GPU memory.

```bash
git clone https://github.com/moghbaie/supply-chain-agent.git
cd supply-chain-agent
uv venv --python 3.12.10
source .venv/Scripts/activate   # Git Bash on Windows
uv pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
```

Create a file named `.env.secrets` in the repo root. It is ignored by git, so never commit it.

```text
HF_TOKEN=
KAGGLE_USERNAME=
KAGGLE_KEY=
TABPFN_TOKEN=
```

Only `scripts/download_data.py` and `scripts/check_api_keys.py` read this file. For model downloads, also set `HF_TOKEN` and `TABPFN_TOKEN` in your shell. Kumo and Jev keys are only needed by the saved code from Steps 6 and 7, and sending to both is off. You also need to accept the M5 competition terms on Kaggle and the TabPFN license.

## Running it

Data and model outputs are not in git, so each script rebuilds its inputs from earlier steps. Run these from the repo root:

```bash
python scripts/download_data.py              # M5 and Olist
python scripts/run_simulation.py             # inventory simulator
python scripts/run_forecasting.py            # forecasting benchmark
python scripts/build_stockout_forecasts.py   # forecasts on simulated sales
python scripts/build_stockout_features.py
python scripts/run_stockout.py               # TabPFN and baselines
python scripts/run_delay_baselines.py        # delay-risk tables and baselines
```

**Known gap:** a fresh clone cannot currently run past this point without changes. `scripts/build_agent_data.py` reads `classified_test.parquet`, which `scripts/run_escalation.py` builds alongside its Jev calls. Sending to Jev is off and the saved answers are not in git, so the escalation step needs either those files or a version without Jev.

With the Step 7 outputs in place, the rest runs as follows:

```bash
python scripts/run_ultrabert.py              # fine-tune and score the classifier
python scripts/build_agent_data.py           # agent cases and delay scores
python scripts/run_agent.py --cases test --per-category 25
python scripts/score_agent.py
python scripts/run_replay.py                 # Step 10 replay; resumes if interrupted
python scripts/score_replay.py
python scripts/build_replay_sample.py        # the cases for the LLM sample
python scripts/run_agent.py --cases replay --conflict-rule
python scripts/score_llm_sample.py
python scripts/build_demo.py                 # the results page, reports/step_10/demo.html
```

The full replay takes about 6 hours on this hardware, and the 100-case LLM sample about 2.2 hours. To try a short run that saves nothing, use `python scripts/run_replay.py --scenarios baseline --variants reorder_point --last-day 30`.

To run the tests:

```bash
python -m pytest
```

Tests read the local data and model outputs, so run them after the steps above. No test calls a hosted API. The full suite last passed at Step 10 (469 of 469).

## Repository layout

```text
configs/    simulator, forecasting, agent policy and evaluation settings
scripts/    entry points for each step, plus inspection and probe scripts
src/
  simulation/   inventory simulator
  forecasting/  demand panel, Timestral runner and metrics
  stockout/     features, TabPFN and baselines
  supplier/     delay features, Olist and the Kumo client (sending off)
  escalation/   cases, policy rule, UltraBERT and the Jev client (sending off)
  agent/        tools, LLM, orchestration, policy checks and explanation audit
  evaluation/   closed-loop replay, model scoring and simulated approvals
tests/      unit, integration and evaluation tests
reports/    one folder per step, with a summary and saved results
```

## Departures from the plan

| Departure | Reason |
|---|---|
| LightGBM replaces Kumo for delay risk | Slightly higher PR-AUC on both datasets, though neither gap was clear, and it runs locally |
| Fine-tuned UltraBERT replaces Jev for escalation | Clearly better than Jev on macro-F1 and critical recall, and it runs locally |
| SmolLM3-3B replaces Qwen3-4B | Qwen3-4B finished none of 3 cases in 10 minutes on this GPU |
| Step 9 skipped | The agent stays untuned, and Step 10 evaluates it as is |

## Limitations

- All inventory results come from one simulator, so they say nothing direct about real supply chains.
- The escalation categories are hindsight definitions written for this project, not expert labels.
- The policy passages the agent can search were written for this project, not taken from a real company.
- The Step 10 replay simulates the person who approves actions, and the time saved by expediting and the expedite fee are assumptions. Results are sensitive to these.

## License

This repository has no license file yet, so no reuse rights are granted by default. Datasets and models keep their own terms.
