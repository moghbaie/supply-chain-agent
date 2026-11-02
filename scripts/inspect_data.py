"""Print and save schemas, previews, counts, missingness, dates, duplicates and key coverage for one dataset."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
pd.set_option("display.width", 170, "display.max_columns", 12, "display.max_colwidth", 22)

KEYS = {
    "olist": {
        "olist_customers_dataset": ["customer_id"],
        "olist_orders_dataset": ["order_id"],
        "olist_order_items_dataset": ["order_id", "order_item_id"],
        "olist_order_payments_dataset": ["order_id", "payment_sequential"],
        "olist_order_reviews_dataset": ["review_id", "order_id"],
        "olist_products_dataset": ["product_id"],
        "olist_sellers_dataset": ["seller_id"],
        "product_category_name_translation": ["product_category_name"],
    },
    "m5": {
        "calendar": ["d"],
        "sell_prices": ["store_id", "item_id", "wm_yr_wk"],
        "sales_train_validation": ["id"],
        "sales_train_evaluation": ["id"],
    },
}
FOREIGN_KEYS = {
    "olist": [
        ("olist_orders_dataset", "customer_id", "olist_customers_dataset", "customer_id"),
        ("olist_order_items_dataset", "order_id", "olist_orders_dataset", "order_id"),
        ("olist_order_items_dataset", "product_id", "olist_products_dataset", "product_id"),
        ("olist_order_items_dataset", "seller_id", "olist_sellers_dataset", "seller_id"),
        ("olist_order_payments_dataset", "order_id", "olist_orders_dataset", "order_id"),
        ("olist_order_reviews_dataset", "order_id", "olist_orders_dataset", "order_id"),
        ("olist_products_dataset", "product_category_name", "product_category_name_translation", "product_category_name"),
        ("olist_customers_dataset", "customer_zip_code_prefix", "olist_geolocation_dataset", "geolocation_zip_code_prefix"),
        ("olist_sellers_dataset", "seller_zip_code_prefix", "olist_geolocation_dataset", "geolocation_zip_code_prefix"),
    ],
    "m5": [
        ("sell_prices", "wm_yr_wk", "calendar", "wm_yr_wk"),
        ("sales_train_evaluation", "item_id", "sell_prices", "item_id"),
        ("sales_train_evaluation", "store_id", "sell_prices", "store_id"),
    ],
}

dataset = sys.argv[1]
folder = Path("data/raw") / dataset
out = Path("reports/step_02") / dataset
out.mkdir(parents=True, exist_ok=True)
tables = {p.stem: pd.read_csv(p, low_memory=False) for p in sorted(folder.glob("*.csv"))}
report = {"tables": {}, "foreign_keys": []}

for name, df in tables.items():
    date_columns = [c for c in df.columns if "date" in c.lower() or "timestamp" in c.lower()]
    dates = {}
    for column in date_columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        dates[column] = {"min": str(parsed.min()), "max": str(parsed.max()), "unparsed_nonnull": int((parsed.isna() & df[column].notna()).sum())}
    key = KEYS.get(dataset, {}).get(name)
    missing = df.isna().sum()
    missing = {c: {"count": int(n), "pct": round(100 * n / len(df), 2)} for c, n in missing.items() if n}
    info = {
        "rows": len(df), "columns": df.shape[1],
        "dtypes": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "key": key,
        "key_duplicates": int(df.duplicated(key).sum()) if key else None,
        "missing": missing, "dates": dates,
    }
    report["tables"][name] = info
    df.head(20).to_csv(out / f"{name}_preview.csv", index=False)
    print(f"\n===== {name}: {len(df):,} rows, {df.shape[1]} columns")
    print("schema:", {c: str(t) for c, t in list(df.dtypes.items())[:14]}, "..." if df.shape[1] > 14 else "")
    print(f"exact duplicates: {info['exact_duplicate_rows']:,} | key {key}: {info['key_duplicates']} duplicates")
    print("missing:", {c: f"{m['count']:,} ({m['pct']}%)" for c, m in list(missing.items())[:14]} or "none")
    if dates:
        print("dates:", dates)
    print(df.iloc[:10, :9].to_string())

print("\n===== relational coverage")
for child, child_col, parent, parent_col in FOREIGN_KEYS.get(dataset, []):
    if child not in tables or parent not in tables:
        continue
    values = tables[child][child_col].dropna()
    unique = values.drop_duplicates()
    matched = unique.isin(tables[parent][parent_col])
    row_match = values.isin(tables[parent][parent_col])
    item = {"child": f"{child}.{child_col}", "parent": f"{parent}.{parent_col}", "unique_values": len(unique),
            "unmatched_unique": int((~matched).sum()), "row_coverage_pct": round(100 * row_match.mean(), 3)}
    report["foreign_keys"].append(item)
    print(f"{item['child']:58} -> {item['parent']:58} coverage {item['row_coverage_pct']}% | unmatched unique {item['unmatched_unique']:,}")

(out / "inspection.json").write_text(json.dumps(report, indent=2))
