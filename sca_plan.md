# Supply Chain Reasoning Agent: Implementation Plan

## Objective
Build and fine-tune a reasoning LLM that coordinates specialized models to forecast demand, estimate stockout risk, assess supplier delays, classify escalations, and recommend inventory actions. The system must preserve human approval for purchasing decisions.

Proposed components:
- Timestral: 14–28-day demand forecasts, subject to checkpoint and interface verification.
- TabPFN: 14-day stockout probability using point-in-time inventory and forecast features.
- Kumo Relational: relational delivery-risk prediction, with inbound supplier-delay modeling evaluated separately.
- Jev: predefined escalation category classification, subject to interface verification.
- Reasoning LLM: plans investigations, calls tools, interprets evidence, and recommends actions.
- Deterministic policy engine: checks constraints and approval requirements.

## Mandatory workflow for every step
1. State the step's objective, inputs, outputs, dependencies, and acceptance criteria.
2. Implement **only the current step**.
3. Execute the implementation and **print actual data**: schemas, first 10–20 rows, counts, missingness, date ranges, and distributions where relevant.
4. **Show actual results**: calculations, predictions, metrics, baselines, diagnostic plots, runtime and resource usage where relevant.
5. **Only after inspecting the data and results**, write automated tests for the step.
6. Run tests and print each test name, PASS/FAIL, and failure details. Fix failures, rerun the step, reprint changed results, and rerun tests.
7. Save reproducible artifacts to `reports/step_XX/`: previews, metrics, plots, logs, config, seeds, versions, and review summary.
8. **Stop and request explicit approval** before proceeding to the next step, even if all tests pass.

Never invent results or claim tests passed without executing them. Distinguish observed and synthetic data. Do not substitute a specialized model without documenting the reason and requesting approval. Use chronological splits and enforce point-in-time correctness.

## Step-by-step roadmap

### Step 1. Environment and repository validation
- Inspect AirLLM and Soup, pin commit hashes, inspect licenses, and establish a reproducible Python environment.
- Record Python, PyTorch, GPU/VRAM, RAM, disk, CUDA and dependency versions.
- Print: environment report, repository metadata, installed packages and available compute.
- Tests: imports, dependency compatibility, minimal model loading and deterministic seeding.
- Gate: reproducible environment and verified training/inference tool selection.

### Step 2. Download and inspect public datasets
- Download M5 forecasting data and Olist Brazilian E-Commerce data into separate raw directories.
- Identify SKU, sales, dates, sellers, orders, shipments and delivery fields.
- Print: first 10 rows per table, schemas, row counts, missingness, date ranges, duplicates and relational coverage.
- Tests: unique keys, foreign keys, chronological consistency and required-field completeness.
- Gate: dataset limitations documented, especially missing real inventory and inbound purchase-order history.

Sources:
- M5: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
- Olist: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Important: M5 and Olist are unrelated companies and must not be joined as if they describe the same business. Review dataset licenses before commercial use.

### Step 3. Build a coherent inventory simulation
- Use M5 demand with synthetic inventory snapshots, suppliers, replenishment purchase orders, lead times, receipts and delays.
- Define inventory conservation, unmet demand, and stockout labels.
- Keep Olist as a separate real-data delivery-risk benchmark.
- Print: 20 inventory records, stock trajectories, lead-time distributions, generated stockout events and scenario parameters.
- Tests: conservation, nonnegative physical stock, chronology, reproducibility and known edge cases.
- Gate: internally consistent, auditable simulator with train/validation/test periods.

### Step 4. Implement demand forecasting
- Verify the exact Timestral model and supported forecasting interface before use. If unsuitable, propose and seek approval for a substitute.
- Benchmark against seasonal naive and an appropriate classical/boosted baseline.
- Print: actual vs predicted demand for 10 SKUs, 14–28-day forecasts, uncertainty intervals, WAPE and bias.
- Tests: no future leakage, horizon alignment, valid intervals, stable inference and baseline comparison.
- Gate: documented forecast performance on unseen time periods.

### Step 5. Train TabPFN stockout-risk model
- Create point-in-time features from on-hand inventory, inbound orders, forecasts, lead times and supplier history.
- Label stockouts within the next 14 days, without leaking future information into inputs.
- Print: feature schema, 20 rows, class balance, 20 predictions, PR-AUC, Brier score and calibration plots.
- Tests: label correctness, feature availability, chronological splits, missing values and class imbalance.
- Gate: calibrated, reproducible predictions against a simple baseline.

### Step 6. Implement Kumo Relational delay-risk modeling
- Create connected orders, sellers, shipments and delivery tables.
- Benchmark late customer-delivery prediction on Olist; separately model synthetic inbound supplier delays.
- Print: relationship schema, example joins, delay predictions, PR-AUC and calibration.
- Tests: relationship integrity, point-in-time joins, unseen-entity behavior and leakage prevention.
- Gate: clear distinction between customer-delivery risk and supplier-replenishment risk.

### Step 7. Implement Jev escalation classification
- Verify Jev's availability, checkpoint, license and classification interface.
- Define categories: monitor, reorder recommendation, expedite recommendation and human review.
- Compare to a deterministic policy baseline.
- Print: 20 classified cases, counts, confusion matrix, macro-F1 and critical escalation recall.
- Tests: category validity, conflicting signals, critical-case handling and policy overrides.
- Gate: stable escalation taxonomy and acceptable safety performance.

### Step 8. Build an untuned reasoning-agent baseline
- Select a 3B–8B instruction model and implement typed tools, orchestration, retrieval and policy checks.
- Make tools callable independently where appropriate; consolidate results before final decisions.
- Print: 10 complete agent traces, tool inputs/outputs, explanations, decisions, tokens, latency and costs.
- Tests: tool contracts, missing data, failures, unsupported claims, invalid actions and approval boundaries.
- Gate: baseline performance recorded before fine-tuning.

### Step 9. Fine-tune the reasoning LLM
- Generate 2,000–5,000 initial supervised examples from simulator states, tool outputs and independently checked reference decisions.
- Include tool-call structure, evidence interpretation, uncertainty and escalation; do not train on hidden future information.
- Compare Soup-based LoRA/QLoRA with standard PEFT and the untouched baseline, using held-out scenario families.
- Print: sample examples, training/validation loss, throughput, VRAM/RAM, checkpoint metadata and before/after performance.
- Tests: dataset separation, label validity, checkpoint reload, reproducibility and regression tests.
- Gate: measurable improvement over the untuned agent, without degraded safety.

### Step 10. End-to-end evaluation and demo
- Replay unseen historical and synthetic scenarios against reorder-point and forecast-driven baselines under identical constraints.
- Print: stockout rate, unmet demand, holding cost, expedite cost, total cost, review workload, errors and example recommendations.
- Tests: integration, recovery, policy compliance, regression and stress cases.
- Gate: reproducible end-to-end results with explicit limitations and human approval for purchases.

## Standard step report
Every step must produce `reports/step_XX/summary.md` with:
- Objective and implementation completed
- Data inspected, schemas, preview rows and quality issues
- Results, metrics, baselines and plots
- Tests executed, individual PASS/FAIL results and failures fixed
- Limitations and unresolved risks
- Acceptance criteria assessment
- **STOP: Awaiting user approval to proceed**

## Suggested project layout
```text
supply-chain-agent/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── simulated/
├── src/
│   ├── data/
│   ├── simulation/
│   ├── forecasting/
│   ├── stockout/
│   ├── supplier/
│   ├── escalation/
│   ├── agent/
│   └── training/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
├── reports/
├── notebooks/
├── scripts/
├── pyproject.toml
└── README.md
```

## Coding-agent instruction
Implement this plan one step at a time. Before each step, state objectives and acceptance criteria. After implementation, run the code, print real input data and computed results, inspect them, then write and execute tests. Print individual test outcomes, fix failures and rerun. Save artifacts in `reports/step_XX/`. Never fabricate metrics or test outcomes. Never move to the next step without explicit user approval. **Begin with Step 1 only.**
