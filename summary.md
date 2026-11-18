# Step 11: Claim-level explanation judge

**Status: code layer complete and tested on the 10 committed traces. LLM layer built and tested with a scripted model, not yet run. Waiting for approval.** All cases are simulator data. Nothing left the machine; no API was called.

## Objective

Step 8 found that 35 of 100 explanations break at least one narrow rule, and said the count is a lower bound. This step replaces whole-explanation rules with a check on each claim: split the explanation into claims, tie each claim to the evidence field it describes, and say whether it is supported.

| | |
|---|---|
| Inputs | Agent explanations and the tool results each run saw (Step 8 traces) |
| Outputs | One row per claim with a verdict; one row per explanation; a file for hand labels |
| Depends on | Step 8 traces, `src/agent/review.py` (number pattern), `src/agent/audit.py` (for comparison) |
| Acceptance | (1) Reproduces the Step 8 audit input exactly. (2) Finds every number or open-order error the Step 8 rules find. (3) Binds at least 95% of stated numbers to a named field. (4) The LLM judge is checked against hand labels before any LLM count is reported. |

## What was built

| Part | File | Notes |
|---|---|---|
| Judge | `src/agent/judge.py` | Claim splitting, number-to-field binding, code verdicts, LLM judge interface |
| Trace parser | `scripts/parse_trace_file.py` | Rebuilds evidence for the 10 complete traces in `reports/step_08/test_traces.txt`, because full traces are git-ignored |
| Run | `scripts/run_judge.py` | `--source traces` (all 100, local) or `report` (10 committed); `--llm none/qwen/smollm`; `--orders K` for position bias |
| Coverage | `scripts/judge_coverage.py` | Binding coverage on all 100 explanations; needs no evidence |
| Agreement | `scripts/judge_agreement.py` | Kappa, precision and recall against hand labels |
| Tests | `tests/evaluation/test_judge.py` | 50 tests |

### How a claim is checked

| Claim kind | Checked by | Verdicts |
|---|---|---|
| Number ("delay risk of 5.2%") | Code: bind the number to the field named next to it, then compare | supported, wrong_value, misattributed (the value belongs to another field), unsupported (in no tool result), unverified (in the evidence, field not named) |
| Open orders ("there are no open orders", "orders are delayed") | Code | supported, contradicted |
| Value band ("a low value band") | Code | supported, contradicted |
| Qualitative ("low risk of stockout", "late rates are not a concern") | LLM judge, with the evidence and the project's definitions in the prompt | supported, contradicted, not_checked; judge_error for a reply that is not valid JSON |

Rules I set before looking at results:
- A stated whole number may round or truncate the actual value (within 1). A stated decimal must match to its last digit (within 0.051).
- Only a decimal can be called misattributed. A whole number can equal another field by chance; Step 11's own test caught "5 units" matching a 5.0% probability.
- "next 14 days", "last 30 days" and "last 28 days" are horizons, not claims.
- The LLM judge sees one claim at a time, never the whole explanation, and never the agent's category.
- The default judge model is Qwen3-4B, not SmolLM3, so the judge does not share the agent's model family.

## Data check

`parse_trace_file.py` found 10 complete traces, each with all 5 tool results, and all 100 explanations. Running the Step 8 audit on the parsed evidence gives the saved `explanation_audit.csv` result on all 10 cases.

## Results: 10 committed traces, code layer only

80 claims in 48 clauses.

| Kind | Supported | Wrong value | Misattributed | Contradicted | Not checked | All |
|---|---:|---:|---:|---:|---:|---:|
| Number | 51 | 3 | 2 | | | 56 |
| Open orders | 7 | | | 3 | | 10 |
| Value band | 1 | | | | | 1 |
| Qualitative | | | | | 13 | 13 |

**Explanations with a factual error: 7 of 10 by the judge, 5 by the Step 8 rules.**

| Case | Final | Judge finds | Step 8 rules | Live check |
|---|---|---|---|---|
| test-0001 | human review | Stockout risk "20%"; both models say 0.2% | none | flagged 20 |
| test-0002 | monitor | "No open orders"; one is 2 days overdue | same | passed |
| test-0003 | reorder | "Stockout risk of 67.7%" is UltraBERT's reorder probability; models say 3.1% and 3.9% | same | passed |
| test-0004 | human review | (qualitative; needs the LLM) | says it conflicts with the suggestion it followed | passed |
| test-0006 | expedite | "Open orders are delayed"; nothing overdue | same | passed |
| test-0007 | reorder | "37.5% and 41.5% over the last 30 days"; 37.5% is the long-run rate | none | passed |
| test-0041 | human review | "1.0 (100%)"; LightGBM says 1.0% | none | flagged 100 |
| test-0087 | reorder | Stockout risk "70%" (models say 0.7% and 0.5%); describes open orders that do not exist | same | flagged 70 |

- The judge finds every number and open-order error the rules find. The one rule hit it misses, test-0004, is a qualitative claim left for the LLM.
- test-0007 is caught by no earlier check. Every number in it exists in the evidence, so the live check passes it; only binding the number to its field shows the error.
- test-0002 is still the only error on a final monitor case, which no person reviews.

## Results: coverage on all 100 explanations

This needs no evidence, so it says nothing about whether claims are true.

| Measure | Result |
|---|---|
| Clauses | 432, median 4 per explanation |
| Numbers stated | 443 |
| Bound to a named field | 437 (98.6%) |
| Open-order claims checked by code | 94 |
| Clauses sent to the LLM judge | 185, in 95 explanations |

The 6 unbound numbers are in `coverage.log`, such as "the medium value band of 18 units" (test-0012), which Step 8 noted as an error no rule covers. An unbound number is still checked against every tool result.

## Tests

`tests/evaluation/test_judge.py`: 50 passed (`tests_judge.log`). Covered: parsing, sentence splitting around decimals, horizons, 15 binding patterns from real explanations, number verdicts and tolerances, missing fields, open orders, value band, the 10 traces, errors the live check misses, the LLM interface with a scripted model, strict reply parsing, prompt content, position-bias reordering, agreement scoring, and a coverage regression guard.

Three tests failed on the first run. One found a real problem (a whole number called misattributed by chance), fixed in `check_number`. Two had wrong expected values in the test (18 is within rounding of 17.8; 5.2 is absent once orders are removed), fixed in the test.

The Step 8 agent tests: 77 passed, 2 failed in this environment because the parquet data is not present. No Step 8 code changed.

## Not done yet: needs the GPU machine

1. Run on all 100 traces, code layer: `python scripts/run_judge.py --source traces --llm none`
2. Add the LLM judge with three orderings: `python scripts/run_judge.py --source traces --llm qwen --orders 3`. This prints how many qualitative verdicts change when only the order of tool results changes.
3. Hand-label claims in `claims_to_label_traces_qwen.csv` (hide the judge's verdict column while labelling), then `python scripts/judge_agreement.py --tag traces_qwen`. Report LLM counts only after this.

Suggested labelling sample: all claims the judge flags, plus a random 100 of the rest, so both precision and recall are measured.

## Limits

- The code layer is checked on 10 explanations here. The binding rules were written while reading these 10 and the 100 explanation texts, so the all-100 run is the first test on evidence it was not tuned on, and it needs hand labels.
- The qualitative definitions in the judge prompt (high risk at 50% or more, low under 20%, late rates under 10% are low) come from the Step 8 audit limits and the policy file. They are project choices, not measured thresholds.
- No LLM verdict has been produced yet, so nothing here measures the LLM judge.

## Acceptance

| Criterion | Result |
|---|---|
| Reproduces Step 8 audit input | Met, 10 of 10 |
| Finds every number or open-order error the rules find | Met on the 10 traces |
| At least 95% of numbers bound | Met, 98.6% |
| LLM judge checked against hand labels | Not yet; needs the GPU run and labels |

**STOP: Awaiting user approval to proceed.**
