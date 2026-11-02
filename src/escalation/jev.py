"""Ask Jev to choose an escalation category directly from facts computed in code.

Requests carry no item names, IDs, raw prices, dates or scenario names. All question wording lives here.
"""
import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://api.typesafe.ai/v1/systemone"
SENDING_ENABLED = False
MODEL = "jev-1.14.0"
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000
INSTRUCTIONS = "Which outcome fits this item's weekly inventory review, using `item`?"
CRITERIA = {
    "monitor": "Stock looks sufficient for the next 14 days, so no action is needed.",
    "reorder_recommendation": "A stockout looks likely in the next 14 days and no order is on the way.",
    "expedite_recommendation": "A stockout looks likely in the next 14 days and an order is on the way but may arrive too late.",
    "human_review": "A stockout looks likely in the next 14 days and would be costly, or the signals conflict, so a person should decide.",
}
FORBIDDEN_TERMS = ["FOODS", "HOBBIES", "HOUSEHOLD", "_evaluation", "SUP_", "baseline", "lean_policy", "supplier_disruption", "olist", "tier", "price", "future_lost_margin", "reference"]


def number(value, digits=1):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def case_state(row):
    """Return what a planner can see, with comparisons already worked out."""
    days_to_due = number(row.days_to_next_due, 0)
    cover = number(row.cover_days)
    lasts = None if days_to_due is None else bool(row.cover_days >= row.days_to_next_due)
    return {"item": {
        "stock": {"units_on_hand": int(row.on_hand), "days_of_stock_at_forecast_demand": cover, "units_on_order": int(row.on_order), "open_orders": int(row.open_pos),
                  "order_placed_at_this_review": bool(row.ordered_today > 0)},
        "forecast_next_14_days": {"expected_units": number(row.fc_mean_14), "high_estimate_units": number(row.fc_q90_sum_14)},
        "inbound": {"days_until_next_promised_delivery": days_to_due, "units_promised_within_14_days": number(row.units_due_14, 0), "units_overdue": number(row.units_overdue, 0),
                    "most_days_overdue": number(row.days_overdue_max, 0), "stock_lasts_until_next_delivery": lasts},
        "supplier_late_delivery_percent": {"long_run": number(100 * row.supplier_late_rate, 0), "last_30_days": number(100 * row.supplier_recent_late_rate, 0)},
        "stockout_chance_percent_next_14_days": {"primary_model": number(100 * row.stockout_risk, 0), "second_model": number(100 * row.stockout_risk_lightgbm, 0)},
        "sales_value_band": row.value_band}}


def build_request(row, criteria=None):
    return {"model": MODEL, "state": case_state(row), "questions": {"category": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria or CRITERIA}}}


def assert_clean(body):
    text = json.dumps(body["state"])
    for term in FORBIDDEN_TERMS:
        if term in text:
            raise ValueError(f"forbidden term in request: {term}")


def send(body, key, attempts=5):
    """Post a request, retrying rate limits and server errors, and surface other failures as errors."""
    if not SENDING_ENABLED:
        raise RuntimeError("Sending data to Jev is turned off by the user's decision; use the saved Step 7 answers.")
    data = json.dumps(body).encode()
    for attempt in range(attempts):
        request = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            text = error.read(300).decode(errors="replace")
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise RuntimeError(f"HTTP {error.code}: {text}") from None
            time.sleep(min(2 ** attempt, 30))
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 30))


def classify(cases, key, workers=8, criteria=None, on_progress=None):
    """Return choice, probabilities and input tokens for each case, in order."""
    rows = list(cases.itertuples(index=False))
    bodies = [build_request(row, criteria) for row in rows]
    for body in bodies:
        assert_clean(body)

    def run(body):
        response = send(body, key)
        answer = response["answers"]["category"]
        return answer["choice"], answer["probabilities"], response["usage"]["input_tokens"]

    results = []
    with ThreadPoolExecutor(workers) as pool:
        for index, result in enumerate(pool.map(run, bodies), 1):
            results.append(result)
            if on_progress and index % 500 == 0:
                on_progress(index)
    return results
