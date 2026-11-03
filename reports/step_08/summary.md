# Step 8: Untuned agent baseline

**Status: complete, waiting for approval.** The baseline is recorded. All test cases are simulator data, not observed data. Nothing left this machine in this step except model downloads from Hugging Face.

## Main result

- **The LLM added no independent decision.** SmolLM3-3B picked the same category as fine-tuned UltraBERT on 100 of 100 test cases, as it did on 40 of 40 development cases.
- **The code checks changed 2 of 100 results.** Both moved monitor to human review because the LLM misstated a number in its explanation (0.2% as 20%, and 1.0% as 100%). Neither matched the reference category.
- **Explanations are unreliable.** 35 of 100 broke at least one narrow factual rule, and 16 cited no numbers.
- **The approval boundary held.** No orders were placed, every action needed approval, and all 100 runs finished with a valid recommendation.

## What was built

| Part | File | Notes |
|---|---|---|
| Typed tools | `src/agent/tools.py` | 7 tools, unknown fields refused, built from an allowed column list, anonymous case IDs |
| Model | `src/agent/llm.py` | SmolLM3-3B, 3.08B parameters, 4-bit, runs locally, thinking off, greedy |
| Orchestration | `src/agent/runner.py` | Up to 8 turns; a run that does not finish goes to a person |
| Retrieval | `src/agent/retrieval.py`, `configs/agent_policy.yaml` | BM25 over 10 passages written for this project, not a real company's policy |
| Policy checks | `src/agent/review.py` | Run in code after the model |
| Explanation audit | `src/agent/audit.py` | Run after the test, not part of the live agent |

## Test run

100 cases, 25 per reference category, drawn from the 200 built; 87 series; review dates 2016-01-04 to 2016-05-02. The agent's code did not change after this run started.

| Measure, same 100 cases | Agent final | UltraBERT | Policy | Jev (saved) |
|---|---:|---:|---:|---:|
| Macro-F1, balanced as run | 0.686 | 0.695 | 0.503 | 0.398 |
| Accuracy, balanced as run | 0.70 | 0.71 | 0.55 | 0.45 |
| Macro-F1, reweighted to the real test mix | 0.425 | 0.478 | 0.546 | 0.462 |
| Critical recall (25 cases) | 0.36 | 0.36 | 0.28 | 0.24 |
| False alarms on true monitor (25 cases) | 24% | 20% | 8% | 8% |
| Estimated actions per 27,000 cases | 8,349 | 7,368 | 4,080 | 4,014 |

UltraBERT with the Step 7 overrides gives the same results as UltraBERT here, and no override fired for the agent. Reweighted figures rest on only 25 cases per category, so they are rough.

### Differences, series bootstrap, 1,000 rounds [2.5%, 97.5%]

| Agent minus | Macro-F1 balanced | Macro-F1 real mix | Critical recall | False alarms |
|---|---|---|---|---|
| UltraBERT | -0.029, 0.004 | -0.113, 0.000 | 0.000, 0.000 | 0.000, 0.131 |
| Policy | **0.067, 0.307** | -0.215, 0.091 | -0.154, 0.318 | **0.036, 0.318** |
| Jev (saved) | **0.165, 0.424** | -0.145, 0.174 | -0.095, 0.375 | -0.030, 0.350 |

Bold intervals exclude zero. The agent beats the policy and Jev on balanced macro-F1 and raises more false alarms than the policy. Other gaps are not clear at this sample size.

### Confusion, agent final (rows reference)

| | monitor | reorder | expedite | human |
|---|---:|---:|---:|---:|
| monitor | 19 | 3 | 2 | 1 |
| reorder | 2 | 22 | 0 | 1 |
| expedite | 2 | 0 | 20 | 3 |
| human | 1 | 3 | 12 | 9 |

### Actual stockout in the next 14 days, by final category

This uses a label the agent never saw, on the balanced sample.

| Final category | Cases | Stocked out |
|---|---:|---:|
| expedite | 34 | 94% |
| human review | 14 | 93% |
| reorder | 28 | 89% |
| monitor | 24 | 21% |

## Explanation audit

I read all 100 explanations. `scripts/audit_explanations.py` then counted errors with narrow rules. The counts are a lower bound: I spot-checked the flagged cases but did not hand-label all 100, and my reading found other errors no rule covers, such as "the medium value band of 18 units".

| Rule broken | Cases |
|---|---:|
| Any rule | 42 |
| Any factual rule (all except "cites no numbers") | 35 |
| Cites no numbers | 16 |
| Stockout risk number from neither stockout model | 10 |
| Says high risk, both models under 20% | 10 |
| Says it conflicts with the suggestion it followed | 7 |
| Describes an open order that does not exist | 4 |
| Says no open orders, one exists | 3 |
| Says delayed, nothing overdue | 3 |

- **Unreviewed errors:** 5 of the 24 final monitor cases have a factual error, and monitor needs no approval, so no person would see them. Example: test-0002 says "There are no open orders" when an order is 2 days overdue.
- **What the live check misses:** the numeric check only asks whether a number appeared in any tool result. Example: test-0003 calls UltraBERT's 67.7% a stockout risk when the models say 3.1% and 3.9%, and it passes.
- **Boilerplate:** 7 explanations are word for word the same.
- **Uncertainty field:** "high" on 61 of 100, including cases where the explanation calls the suggestion high probability, so it carries little information.

## Other findings

- **Policy conflict rule not enforced.** The policy says to treat a gap of 30 points or more between the stockout models as conflicting and say so, and the category definition sends conflicting evidence to human review. Four test cases had such a gap (test-0007, 0018, 0054, 0095). None went to human review, and code does not check this. I did not change it after seeing test results.
- **Retrieval went unused.** `search_policy` had 0 calls in the 100 test cases and 40 development cases.
- **Weak evidence.** 14 actions had both stockout risks under 10%. 11 were true reorder cases, 2 monitor, 1 human review. All still need approval.
- **Unsupported numbers.** 3 cases. test-0087 stayed reorder with its flag, because only monitor cases are moved to a person for this.

## Run cost

| Measure | Result |
|---|---|
| Seconds per case | Median 86, mean 82, 95th percentile 109, maximum 118 |
| Turns per case | Median 7, mean 6.6 |
| Tokens per case | Mean 8,925 in, 322 out |
| Peak GPU memory | 2.20 GB |
| Total run | 8,156 s (2 h 16 min) |
| API cost | $0; electricity and hardware not measured |
| Failed runs, invalid submissions, bad tool calls, repeat stops | 0 |
| Malformed calls | 61, recovered by the next turn |
| Untagged calls | 49 in 40 cases, accepted by the lenient parser |

At this speed all 27,000 test cases would take about 25 days, so Step 10 cannot run the LLM on every case in the replay without a design decision.

## Traces

`test_traces.txt` has 10 complete traces with tool inputs and outputs, explanations, decisions, tokens, latency and cost, followed by all 100 explanations. The 10 were chosen by rule, not by how they read: the first 2 of each category, then the first 2 more with an unsupported number. All traces are in `data/processed/agent/traces/smollm_untuned/test/`, which git ignores.

## Tests

- `tests/evaluation/test_agent.py`: 79 passed. Per-test results are in `tests_agent.log`.
- Full suite: 357 passed. See `tests_full.log`.
- Covered: tool contracts, missing data, tool failures, unsupported claims, invalid actions, approval boundaries, hidden outcomes and IDs, point-in-time open orders, retrieval, hidden tool refusal, the parser, UltraBERT's two odd reorder edge cases, and agreement between saved traces and current tool output.
- A test also caught a bug in my audit: a fix had silently turned off one rule, which lowered the counts. It was corrected before the numbers above.

## Departures from the plan

| Departure | Decided by |
|---|---|
| Step 9 skipped, so the agent stays untuned | User |
| LightGBM replaces Kumo for delay risk, and Kumo and Jev sending are off | User |
| Fine-tuned UltraBERT is the escalation input | User |
| Qwen3-4B stopped after finishing none of 3 cases in 10 minutes; SmolLM3-3B used | User |
| 100 test cases run out of the 200 built, as the agreed option | User |
| No-suggestion version tried on 4 development cases only, not on test | My recommendation, with no objection from the user |
| Thinking off, for speed | Me |
| Lenient parser accepts Python-style and untagged calls, and counts them | Me |
| Turn limit raised to 8, repeat stop added, shared prompt cache added after a 3-case development trial | Me |
| Windows kept awake during runs, with no settings changed, after a sleep during the development run | Me |

## Limits

- Everything here is one simulator's data.
- Critical recall stays low (0.36 here, 0.41 on the full test set), as already accepted.
- Development scores are not comparable to Step 7, because UltraBERT's epoch was selected on that period.
- The no-suggestion trial was 4 cases, too few to conclude whether the LLM can judge cases alone.
