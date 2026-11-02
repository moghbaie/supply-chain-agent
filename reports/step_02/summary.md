# Step 2 summary: Download and inspect public datasets

## Objective and implementation
Downloaded M5 and Olist into `data/raw/m5/` and `data/raw/olist/`, recorded SHA-256 checksums in `manifest.json`, and inspected both. All figures below are observed data. The datasets were not joined.
Scripts: `download_data.py`, `inspect_data.py`, `inspect_olist_delivery.py`, `inspect_m5_demand.py`.

## M5 (430 MB)
- 30,490 item-store series: 3,049 items, 10 stores, states CA/TX/WI, categories FOODS/HOBBIES/HOUSEHOLD.
- Sales are daily from 2011-01-29 to 2016-05-22 (1,941 days). The calendar runs 28 days further to 2016-06-19 with no sales.
- `sales_train_validation` matches `sales_train_evaluation` on its 1,913 shared days, so evaluation is the file to use.
- sell_prices: 6,841,121 weekly rows, no missing values or duplicate keys.
- 68% of daily cells are zero; 59.53% after each series' first sale. 20.78% of days precede a series' first listed price, and no units sold before it.
- Dec 25 is zero in 99.96% of series, so the stores appear closed.

## Olist (121 MB)
- 9 tables; orders 99,441, items 112,650, sellers 3,095, products 32,951.
- Core foreign keys match at 100%.
- 6,534 of 96,470 delivered orders (6.77%) were late against the estimate. Monthly rate ranges from 1.2% (2018-06) to 19.0% (2018-03).
- Exceptions: 166 carrier dates before purchase, 1,359 before approval, 23 customer dates before carrier, 8 delivered orders with no date, 4 item rows with shipping limits in 2020.

## Limitations and unresolved risks
- Neither dataset has inventory, inbound purchase orders, receipts or lead times; a test confirms the column names contain none.
- M5 records sales, not demand. A zero can mean no demand, a stockout, or an item not carried.
- Zero-run heuristic: 331,722 runs in 28,818 series (31.34% of post-launch zero days) are unlikely under a constant Poisson rate at p<0.001. This is a rough signal, not a stockout count. It assumes constant demand, is uncorrected for multiple comparisons, and includes store closures.
- Some series have zero runs of 1,500+ days despite having listed prices, for example HOUSEHOLD_1_512_CA_4 at 1,770 days.
- Olist is CC BY-NC-SA 4.0, which is non-commercial. M5's competition terms were accepted on Kaggle but have not been read for this report; review before any commercial use.
- Olist reviews and delivery dates occur after the outcome and must not be inputs in Step 6.
- M5 and Olist describe unrelated companies and must not be joined.

## Tests
34 of 34 passed; individual results are in `tests.log`. No failures needed fixing. Some tests pin documented exception counts, so they will flag changed data.

## Acceptance criteria
- Previews, schemas, counts, missingness, date ranges, duplicates and coverage: met; saved per table.
- Unique keys, foreign keys, chronology and required fields tested: met.
- Limitations documented: met.

## STOP: Awaiting user approval to proceed
