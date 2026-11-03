"""Run one Kumo prediction on small synthetic tables; the key comes from KUMO_API_KEY. Disabled, because sending to Kumo is off."""
raise SystemExit("Sending to Kumo is off.")
import os

import numpy as np
import pandas as pd

import kumoai.rfm as rfm

rng = np.random.default_rng(42)
users = pd.DataFrame({"user_id": range(300)})
orders = pd.DataFrame({
    "order_id": range(3000),
    "user_id": rng.integers(0, 300, 3000),
    "order_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, 3000), unit="D"),
})
rfm.init(api_key=os.environ["KUMO_API_KEY"], log_level="ERROR")
graph = rfm.Graph.from_data({"users": users, "orders": orders}, verbose=False)
model = rfm.KumoRFM(graph, verbose=False)
result = model.predict("PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR users.user_id = 1", verbose=False)
print("PREDICTION_OK", list(result.columns), result.iloc[0].to_dict())
