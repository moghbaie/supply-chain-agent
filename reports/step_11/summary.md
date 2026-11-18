# Step 11: Claim-level explanation judge

**Status: code layer built, run on all 100 test traces and tested. LLM layer built and tested with a scripted model only; no LLM verdict exists yet. Waiting for approval.** All cases are simulator data. Nothing left the machine; no API was called.

The design came from `summary.md` in the repository root, which described this step as finished. None of its code was in the repository, so it was written here and every number below comes from these runs. Where they differ from that file, both are shown.

## Objective

Step 8 found that 35 of 100 explanations break at least one narrow rule, and said that count is a lower bound. This step checks each claim instead: split the explanation into claims, bind each stated number to the field it describes, and say whether it is supported.

| | |
|---|---|
| Inputs | Agent explanations and the tool results each run saw (Step 8 traces) |
| Outputs | One row per claim with a verdict; one row per explanation; a blind file for hand labels |
| Depends on | Step 8 traces, `src/agent/review.py` (number pattern), `src/agent/audit.py` (open-order patterns, and for comparison) |
| Acceptance | (1) Reproduces the Step 8 audit input exactly. (2) Finds every number or open-order error the Step 8 rules find. (3) Binds at least 95% of stated numbers to a named field. (4) The LLM judge is checked against hand labels before any LLM count is reported. |

## What was built

| Part | File | Notes |
|---|---|---|
| Judge | `src/agent/judge.py` | Claim splitting, number binding, code verdicts, LLM judge interface |
| Trace parser | `scripts/parse_trace_file.py` | Rebuilds evidence for the 10 complete traces in `reports/step_08/test_traces.txt` |
| Run | `scripts/run_judge.py` | `--source traces` (all 100, local) or `report` (10 committed); `--llm none/qwen/smollm`; `--orders K` for position bias |
| Coverage | `scripts/judge_coverage.py` | Binding coverage from text alone |
| Agreement | `scripts/judge_agreement.py` | Precision, weighted recall and weighted kappa against hand labels |
| Tests | `tests/evaluation/test_judge.py` | 76 tests |

### How a claim is checked

| Claim kind | Checked by | Verdicts |
|---|---|---|
| Number | Code: bind the number to the field it names, then compare | supported, wrong_value, misattributed (equals another field), unsupported (in no tool result, or the named field is missing), unverified (unbound but in the evidence) |
| Open orders ("no open orders", "open orders are delayed", "the open order is due…") | Code | supported, contradicted |
| Value band ("in the low value band") | Code | supported, contradicted |
| Qualitative ("low risk of stockout", "late rates are not a concern") | LLM, one claim at a time | supported, contradicted, unclear; judge_error for a reply that is not strict JSON; not_checked when no LLM runs |

Binding: a number takes its unit from the word after it ("%", "units", "days"), then binds to the nearest field cue with that unit in the same clause. Model names, "long run" and "last 30 days", and category names then narrow the field when attached to the number.

Rules set before results:
- A stated whole number may differ from the actual value by less than 1. A stated decimal must be within 0.051.
- Only a decimal can be misattributed; a whole number can equal another field by chance.
- "next 14 days", "last 30 days", "last 28 days" and "over 14 days" are horizons, not claims.
- The LLM judge gets one claim, the sentence it is from, the definitions and the tool results. It never gets the whole explanation or the agent's category.
- The default judge is Qwen3-4B, a different family from the SmolLM3 agent.

Changes made after seeing results, each for a judge mistake found by reading the flagged claims:

| Change | Case that showed it |
|---|---|
| A model or period name must be attached to its number, with no comma, "and" or "while" between | test-0066: "tabpfn risk of 2.2% and a lightgbm risk of 0.8%" bound 2.2 to LightGBM |
| Underscores read as spaces when finding cues | test-0053: "delay_risk_percent 19.1" |
| ", respectively" stays in the clause it closes | test-0069 |
| "units" before "due" is a units cue, leaving "due in" for the days | test-0065, test-0071: "12 units due" was unbound |
| A value band absorbs only a number written straight after it | test-0064: "low value band with 4.7 units" was unbound |
| Zero is never misattributed | test-0017: "0.0 units" equalled 9 fields |

## Data check

`parse_trace_file.py` found 10 complete traces, each with all 5 tool results, and all 100 explanations. The Step 8 audit on parsed evidence equals the saved `explanation_audit.csv` on 10 of 10, and parsed evidence equals the local JSON traces exactly on 10 of 10 (`parse_trace_file.log`).

## Results: 10 committed traces, code layer

| Kind | Supported | Wrong value | Misattributed | Contradicted | Not checked | All |
|---|---:|---:|---:|---:|---:|---:|
| Number | 51 | 3 | 2 | | | 56 |
| Open orders | 7 | | | 3 | | 10 |
| Value band | 1 | | | | | 1 |
| Qualitative | | | | | 15 | 15 |

**Explanations with a factual error: 7 of 10 by the judge, 5 by the Step 8 rules.** These match the root `summary.md` exactly, except 15 qualitative claims here against its 13, and 55 clauses with a claim against its 48.

| Case | Final | Judge finds | Step 8 rules |
|---|---|---|---|
| test-0001 | human review | Stockout risk "20%"; both models say 0.2% | none |
| test-0002 | monitor | "No open orders"; one is 2 days overdue | same |
| test-0003 | reorder | "Stockout risk of 67.7%" is UltraBERT's reorder probability | same |
| test-0004 | human review | (qualitative; needs the LLM) | says it conflicts with the suggestion it followed |
| test-0006 | expedite | "Open orders are delayed"; nothing overdue | same |
| test-0007 | reorder | "37.5% and 41.5% over the last 30 days"; 37.5% is the long-run rate | none |
| test-0041 | human review | "1.0 (100%)"; LightGBM says 1.0% | none |
| test-0087 | reorder | Stockout risk "70%" (models 0.7%, 0.5%); describes open orders that do not exist | same |

## Results: all 100 traces, code layer

741 claims in 489 clauses.

| Kind | Supported | Wrong value | Misattributed | Contradicted | Unverified | Not checked | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| Number | 421 | 7 | 12 | | 2 | | 442 |
| Open orders | 78 | | | 10 | | | 88 |
| Value band | 17 | | | | | | 17 |
| Qualitative | | | | | | 194 | 194 |

- **27 explanations have a code-layer error; the Step 8 rules flag 35.**
- Every one of the 19 explanations where a Step 8 number or open-order rule fired is also flagged by the code layer.
- The other 16 rule hits are qualitative ("says high risk but both models under 20", "says it conflicts with the suggestion it followed"). They wait for the LLM layer.
- 8 explanations have errors no Step 8 rule found: test-0001, 0007, 0017, 0025, 0041, 0042, 0062, 0082.
- 7 errors are on a final monitor case, which no person reviews: test-0002, 0008, 0025, 0030, 0062, 0063, 0082.
- The most common misattribution: 8 explanations state UltraBERT's category probability as the stockout risk (test-0003, 0014, 0040, 0058, 0072, 0088, 0093, 0112).

Disputed case for the hand labels: test-0062 says "a high probability of 0.8" for monitor. The tool says 75.5%, and 0.8 is the human-review probability. The agent may have meant 0.755 written as a fraction. The judge calls it misattributed.

## Coverage on all 100 explanations

From text alone, so this says nothing about truth (`coverage.log`).

| Measure | Here | Root summary.md |
|---|---|---|
| Clauses | 528, median 5 | 432, median 4 |
| Numbers stated | 442 | 443 |
| Bound to a named field | 440 (99.5%) | 437 (98.6%) |
| Open-order claims | 88 | 94 |
| Clauses sent to the LLM | 194, in 89 explanations | 185, in 95 |

Both unbound numbers follow a value band: "medium value band of 18 units" (test-0012) and "low value band (14 units)" (test-0082). I read 40 randomly chosen bound numbers, and all 40 were bound to the right field. That is my reading, not an independent label.

## Tests

`tests/evaluation/test_judge.py`: 76 passed (`tests_judge.log`). Covered: sentence and clause splitting, horizons, 21 binding patterns from real explanations, tolerances, misattribution rules, "both models", missing fields and tools, open orders, value band, the 10 committed traces, the Step 8 audit reproduction, errors the live check misses, the LLM interface with a scripted model, strict reply parsing, prompt content and tool order, position-bias orders, kappa against scikit-learn, weighted recall, and a coverage guard.

One test failed on the first run, and the test was wrong. It expected "18 units" to be unsupported, but the test evidence had a forecast of 17.8, so 18 is within rounding. The test now uses 99.

The full suite: 545 passed. No Step 8 code changed.

## Not done yet

1. LLM layer: `python scripts/run_judge.py --source traces --llm qwen --orders 3`. This machine has a GTX 1650 (4 GB), which ran SmolLM3 in Step 8, so Qwen3-4B in 4-bit may fit. Not tried yet.
2. Hand labels: fill `label` (error, ok or skip) in `claims_to_label_traces_qwen.csv`. The file has no judge verdict, so labels are blind. Then run `python scripts/judge_agreement.py --tag traces_qwen`. Report LLM counts only after this.

The label sample is every flagged claim plus a random 100 of the rest. Unflagged claims are weighted up to their population for recall and kappa.

## Limits

- The binding rules were written while reading these 100 explanations, and six were changed after seeing results. Coverage and the code-layer counts are therefore measured on the data the rules were tuned on. Hand labels are the only independent check.
- The LLM prompt includes the claim's sentence for context, and that sentence can name the action ("a need for expedited action"). The escalation tool result names the suggested category, which is evidence and often equals the agent's choice.
- The qualitative definitions (high risk at 50% or more, low under 20%, late rates under 10% low) are project choices. The risk limits come from the Step 8 audit. The late-rate limit is not in `configs/agent_policy.yaml`.
- No LLM verdict exists, so nothing here measures the LLM judge.

## Acceptance

| Criterion | Result |
|---|---|
| Reproduces Step 8 audit input | Met, 10 of 10, and parsed evidence equals the local traces |
| Finds every number or open-order error the rules find | Met: 19 of 19 explanations on all 100 traces |
| At least 95% of numbers bound | Met, 99.5% |
| LLM judge checked against hand labels | Not yet; needs the LLM run and labels |

**STOP: Awaiting user approval to proceed.**
