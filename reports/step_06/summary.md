# Step 6 summary: Delay-risk modeling

## Objective and implementation
Benchmark late customer deliveries on real Olist data, and separately model late inbound supplier deliveries on synthetic purchase orders.
Code: `src/supplier/`, `scripts/run_delay_baselines.py`, `scripts/run_kumo_inbound.py`.

The user first declined, then later approved, uploading Olist data to NVIDIA's hosted API. Synthetic purchase orders and Olist were each sent in separate requests and never joined.

## Kumo access
- Kumo Relational is now served through NVIDIA. The `kumoai` default address, `kumorfm.ai/api`, redirects to documentation, which caused the inconclusive Step 1 key check.
- The stored `nvapi-` key works at `https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions`: HTTP 200 for it, HTTP 403 for a fake key.
- Sent: NVIDIA's own sample request (twice, real and fake key), one 105-row synthetic request, 1 refused request, 10 synthetic scoring calls totaling 14.9 MB, one 105-row Olist request, and 20 Olist scoring calls totaling 7.6 MB.
- Olist requests contain only approved columns, with IDs as plain numbers: order, customer state, item (number, seller, product, price, freight), seller state, product (category, weight, dimensions, photos), purchase time, and days until the estimated delivery. The label is sent on example rows only. Actual delivery dates, order status, reviews, payments, cities, zip codes, geolocation and the customer's unique ID are never sent.
- NVIDIA's pages state no data retention, cost or rate limits. Its privacy policy and service terms were not read.

## Olist: customer delivery risk (real data, predicted at purchase)
Relationships: orders to customers, items to orders, items to sellers, items to products. 96,203 delivered orders.
Splits: train 2017 (5.65% late), validation Jan-Mar 2018 (12.88%), test Apr-Aug 2018 (4.42%).

| Model (test, 32,150 orders) | PR-AUC | ROC-AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| Constant rate | 0.0442 | 0.5000 | 0.0425 | 0.0289 |
| Seller's past late rate | 0.0488 | 0.5242 | 0.0430 | 0.0261 |
| LightGBM | 0.0767 | 0.6357 | 0.0426 | 0.0198 |

Orders from sellers with no delivery history: 1,442 (4.49%); LightGBM PR-AUC 0.0817 and Brier 0.0465 on them.

Kumo on the same random 5,000 test orders, with 1,000 labeled examples per call and settings fixed in advance:

| Model (test, same 5,000 orders) | PR-AUC | ROC-AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| Constant rate | 0.0436 | 0.5000 | 0.0420 | 0.0279 |
| Seller's past late rate | 0.0562 | 0.5644 | 0.0424 | 0.0270 |
| LightGBM (60,201 examples) | 0.0729 | 0.6416 | 0.0424 | 0.0213 |
| Kumo (1,000 examples) | 0.0630 | 0.5600 | 0.0424 | 0.0232 |

Kumo minus rival on test, seller bootstrap [2.5%, median, 97.5%]:
- Seller rule: PR-AUC [-0.013, 0.008, 0.029]; Brier [-0.0005, 0.0001, 0.0007].
- LightGBM: PR-AUC [-0.029, -0.011, 0.008]; Brier [-0.0007, 0.0001, 0.0008].

Neither difference is clear. On validation, Kumo's PR-AUC was 0.2500 against LightGBM's 0.2465, and its Brier 0.1160 against 0.1115.

Unseen entities on test: sellers absent from Kumo's 1,000 examples cover 2,396 orders (Kumo PR-AUC 0.073, LightGBM 0.0878). Sellers with no delivered history cover 218 orders, with few late orders (Kumo 0.117, LightGBM 0.0539), which is too small to rely on.

Kumo's predictions stay near 6% each month while actual lateness ranges from 2.00% to 6.90%, and its top bin predicts 14.6% against 7.4% observed. Plot: `kumo_olist_calibration.png`.

## Inbound: supplier delay risk (synthetic data, predicted at order time)
Each order was sent only its supplier's last 15 deliveries received before its date. Kumo settings: 1,000 labeled examples per call (the hosted limit), 500 orders per call, fixed before seeing results. The original 2,000-example setting was refused with HTTP 422.

| Model (test, same 3,000 orders) | PR-AUC | ROC-AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| Constant rate | 0.1723 | 0.5000 | 0.1427 | 0.0413 |
| Supplier's past late rate | 0.3115 | 0.7295 | 0.1277 | 0.0161 |
| LightGBM (72,607 examples) | 0.3355 | 0.7346 | 0.1287 | 0.0351 |
| Kumo (1,000 examples) | 0.3250 | 0.7239 | 0.1300 | 0.0249 |

Kumo minus rival on test, series bootstrap [2.5%, median, 97.5%]:
- Supplier rule: PR-AUC [-0.003, 0.014, 0.032]; Brier [0.0006, 0.0023, 0.0041], so Kumo's Brier is worse.
- LightGBM: PR-AUC [-0.040, -0.012, 0.018]; Brier [-0.0014, 0.0014, 0.0040], so no clear difference.

On supplier disruption, Kumo has the best Brier (0.1534) and ROC-AUC (0.7261) of the three, while LightGBM has the best PR-AUC (0.3838). The sample is small at 831 orders, with no intervals computed per scenario.

## Tests
45 of 45 passed, with no API calls; individual results are in `tests.log`.

## Limitations and unresolved risks
- **Kumo's Olist test is limited.** It saw 1,000 labeled examples per call and no seller delivery history, because past outcomes were not in the approved upload. LightGBM had 60,201 examples and hand-built seller and route history, so this does not show Kumo's best case.
- **The inbound task has little relational structure,** since delays are random draws set by hidden supplier settings.
- **Unseen suppliers and items were not tested with Kumo on inbound calls.** No test order had one. Request building for orders with no history is tested locally.
- **Olist is hard and unstable:** late rates swing from 12.88% to 4.42%, so calibration degrades over time. On validation LightGBM predicted 5.55% on average against 12.88% actual.
- **Olist baselines only cover orders that were delivered.** Cancelled and undelivered orders are excluded, which is a selection effect.
- **Samples:** Kumo scored 5,000 of 35,281 synthetic validation and test orders and 10,000 of 52,777 Olist orders, to limit usage.
- **Test results were visible before Kumo ran,** because baselines were scored on test first. Kumo's settings were fixed in advance and not changed in response, apart from the forced limit of 1,000 examples.
- **The two risks are separate:** Olist measures customer delivery lateness, and inbound measures supplier replenishment lateness. They are never joined, and Olist should not be used to justify supplier risk scores.
- **Inbound results are synthetic.** Supplier behavior comes from assumptions set in Step 3.

## Acceptance criteria
- Relationship schema, example joins, delay predictions, PR-AUC and calibration: met. Kumo's calibration plot is `kumo_calibration.png`; Olist reliability is in `baselines.log`.
- Relationship integrity, point-in-time, unseen-entity and leakage tests: met.
- Clear distinction between customer-delivery and supplier-replenishment risk: met.
- Kumo benchmark on Olist: met, on a 5,000-order test sample, with the limits above.

## STOP: Awaiting user approval to proceed

## Decision after this step
The user approved replacing Kumo with LightGBM as the agent's delay-risk model. This departs from the plan, which names Kumo for this step. LightGBM's PR-AUC was slightly higher on both datasets, with neither gap clear, and it runs locally.
Sending to Kumo is now off in `src/supplier/kumo_client.py`, and `scripts/kumo_probe.py` stops before contacting Kumo. Saved Kumo results stay as a comparison.
