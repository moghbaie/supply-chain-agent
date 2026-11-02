# Step 7 summary: Jev escalation classification

## Objective and implementation
Classify each weekly review case into monitor, reorder recommendation, expedite recommendation or human review, using Jev directly as the user decided, against a deterministic policy baseline.
Code: `src/escalation/`, `scripts/run_escalation.py`, `scripts/inspect_escalation.py`.

**Synthetic data:** every case comes from the Step 3 simulator. No Olist data and no synthetic messages were used. Jev read computed facts only.

## Jev verification
Jev is a hosted API, model `jev-1.14.0`. The stored key worked. The docs state no license or data retention terms on the pages read, and say Jev is not trained on customer requests. The docs' known weak spots page covers version 1.13, so its advice was applied but not confirmed for 1.14.

Requests contained stock, forecast, inbound, supplier and model-risk figures, with no names, IDs, raw prices, dates or scenario names. Comparisons were computed in code. A guard refuses requests with forbidden terms.

Spend: 21,692,527 input tokens, about $0.91 of the $5 limit assumed from the suggestion, since the user approved the step without naming a limit.

## Reference categories
Defined from what happened in the next 14 days, which neither model can see:
- monitor: no stockout.
- human_review: stockout with lost margin of 24.0 or more, the 90th percentile of stockouts in train.
- expedite_recommendation: other stockout, with an order open.
- reorder_recommendation: other stockout, with no order open.

**These are hindsight definitions written for this project, not expert labels or policy.** Test mix: monitor 24,267, expedite 2,196, human review 276, reorder 261.

## Test results (all 27,000 cases)
| Measure | Jev | Policy |
|---|---:|---:|
| Macro-F1 | 0.4284 | 0.4914 |
| Accuracy | 0.8607 | 0.8648 |
| Critical recall (routed to human review) | 0.2355 | 0.3841 |
| Critical cases not left on monitor | 0.9348 | 0.8877 |
| Any stockout acted on | 0.7351 | 0.7530 |
| False alarms among monitor cases | 0.0989 | 0.1083 |
| Human review cases | 1,776 | 406 |
| Recall: monitor | 0.9011 | 0.8917 |
| Recall: reorder | 0.1303 | 0.2261 |
| Recall: expedite | 0.5806 | 0.7040 |

Jev minus policy, series bootstrap [2.5%, median, 97.5%]:
- Macro-F1: [-0.085, -0.063, -0.038].
- Critical recall: [-0.218, -0.148, -0.056].

Policy thresholds were fitted on all validation rows: act at 0.10 risk, critical at 0.90. Jev's wording was not tuned. The policy therefore had a label-based advantage that Jev did not. Confusion matrices: `confusion.png`.

Overrides changed no real case: Jev's results with and without them are identical on test.

## Conflicting signals
- Models disagree by 0.3 or more on 297 test cases. Jev: human review 163, expedite 128, reorder 5, monitor 1. Policy: human review 245, monitor 52.
- Of 10 hand-built edge cases (`edge_cases.csv`), both agree on 6. Where they differ: low risk with 1 day of cover, Jev reorder and policy monitor; high risk with 200 days of cover, Jev human review and policy reorder; models conflicting, Jev reorder and policy human review; high-value high-risk, Jev expedite and policy human review.

## Stability
- Option order: 96.60% same choice on 2,000 test cases, macro-F1 0.4473 against 0.4502. On 500 validation cases, 96.80% same, macro-F1 0.3945 against 0.4628. The rare classes move.
- Repeat calls: 99% same choice on 100 cases.

## Tests
31 of 31 passed; individual results are in `tests.log`. No test calls the API; failures of Jev calls are checked with a stand-in, and raise errors rather than producing a category.

## Limitations and unresolved risks
- **Critical recall is low for both** (0.24 and 0.38). Neither should be relied on to catch severe stockouts alone.
- **No safety threshold was set in advance.** The plan asks for acceptable safety performance and no number was agreed, so this step reports results for a user decision.
- **Severity is hard to see in advance:** the reference uses realized lost margin, while cases show only a value band. Many stockouts are also unpredictable.
- **Jev overuses human review:** 1,776 cases against 276 true ones, which adds workload. Jev's wording says "costly", with no threshold.
- **Direct classification was the user's decision.** TypeSafe's guidance favors judgments from Jev and decisions in code; that was offered in the plan review and declined, and was not tested here.
- **Overrides were only exercised in unit tests** on real data, since none fired.
- **Everything is synthetic:** these results reflect simulator rules and hindsight categories.

## Acceptance criteria
- 20 classified cases, counts, confusion matrix, macro-F1 and critical recall: met.
- Category validity, conflicting signals, critical cases and policy overrides tested: met.
- Stable taxonomy: met; the four categories are fixed and validated.
- Acceptable safety performance: **needs a user decision.**

## STOP: Awaiting user approval to proceed

## Decision after this step
The user chose to keep both the policy and Jev but send no more data to Jev. Sending is now off in `src/escalation/jev.py`. The policy classifies new cases, and Jev is available only through its saved Step 7 answers.
