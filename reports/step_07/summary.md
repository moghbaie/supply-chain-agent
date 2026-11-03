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

## Added option: UltraBERT
The user asked to try UltraBERT (dmis-lab, Apache 2.0) as another option. It ran locally on the 4 GB GPU, so no case data left the machine. Jev and policy results are unchanged, and no Jev calls were made.
Code: `src/escalation/ultrabert.py`, `scripts/run_ultrabert.py`, `scripts/compare_ultrabert.py`. Output: `ultrabert_comparison.log`.

Two routes were tested, each reading the same facts Jev saw, written as a compact sentence:
- **Fine-tuned** (`UltraBERT-150M`): trained on 8,000 cases from 2015 with origins up to 2015-09-14, half of them stockouts, with a weighted loss. Epoch chosen on 3,000 later 2015 cases: epoch 1 of 2, development macro-F1 0.4836. This took 69 minutes. Scores improved up to the last epoch, so more training was not ruled out.
- **Zero-shot** (`UltraBERT-Embed-150M`): no training, picks the category description closest to the case. The descriptions are Jev's wording.

All settings were fixed before the test cases were scored, and no result below was used to change them.

| Measure (all 27,000 test cases) | Jev | Policy | Fine-tuned | Zero-shot |
|---|---:|---:|---:|---:|
| Macro-F1 | 0.4284 | 0.4914 | 0.4922 | 0.1653 |
| Accuracy | 0.8607 | 0.8648 | 0.7891 | 0.2883 |
| Critical recall | 0.2355 | 0.3841 | 0.4130 | 0.0000 |
| Critical cases not left on monitor | 0.9348 | 0.8877 | 0.9565 | 0.9312 |
| Any stockout acted on | 0.7351 | 0.7530 | 0.8383 | 0.9590 |
| False alarms among monitor cases | 0.0989 | 0.1083 | 0.2053 | 0.7731 |
| Recall: reorder | 0.1303 | 0.2261 | 0.8199 | 0.5862 |
| Recall: expedite | 0.5806 | 0.7040 | 0.7705 | 0.9677 |
| Cases with an action | 4,410 | 4,687 | 7,272 | 21,381 |
| Precision of any action | 45.56% | 43.91% | 31.50% | 12.26% |
| Human review cases | 1,776 | 406 | 252 | 2 |

Series bootstrap [2.5%, median, 97.5%], fine-tuned minus each comparison:
- Against policy, macro-F1: [-0.026, -0.000, 0.024]. Critical recall: [-0.081, 0.025, 0.118]. Neither is clearly different.
- Against policy, false alarms: [0.087, 0.097, 0.107]. Any stockout acted on: [0.069, 0.085, 0.102]. Both are clearly higher.
- Against Jev, macro-F1: [0.029, 0.063, 0.096]. Critical recall: [0.032, 0.173, 0.292]. Both are clearly higher.

Zero-shot is clearly worse than both on macro-F1 and critical recall.

**Reading:** fine-tuned UltraBERT beats Jev on these scores but does not beat the policy overall. It trades precision for coverage against the policy: 8.5 points more stockouts acted on for 9.7 points more false alarms, or 2,585 more actions for people to check. Macro-F1 hides this cost, because monitor is 90% of cases but one quarter of the score.

Other observations:
- **Implausible edge cases:** fine-tuned recommends reorder for healthy stock (50 days of cover, 2% risk, probability 0.51) and for zero demand with no stock (0.73). The half-stockout training mix and weighted loss probably push it toward action. That cause was not tested.
- **Not comparable on training:** fine-tuned learned from 2015 outcomes and the policy's thresholds were fitted on them; Jev and zero-shot used no labels.
- **Severe cases:** fine-tuned sends 47% of critical cases to expedite rather than human review, so it still misses most of them as critical.
- **Overrides:** changed no fine-tuned choice. They changed 709 zero-shot choices, moving its macro-F1 to 0.1707.
- **Repeatability:** rescoring 500 test cases gave the same choice on all of them.
- **Tests:** 28 of 28 passed (`tests_ultrabert.log`). None calls an API.
- **Disk:** the fine-tuned weights take 604 MB and stay out of git.

Every choice other than monitor still needs a person's approval, whichever route produces it. **Fine-tuned UltraBERT is not adopted in place of the policy; that is a user decision.**
