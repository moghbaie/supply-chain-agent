"""Step 7 UltraBERT tests: saved scores, category order, what the models could see, and the added results. No test calls an API."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import f1_score

from src.escalation import jev
from src.escalation.cases import CATEGORIES
from src.escalation.jev import CRITERIA
from src.escalation.policy import apply_overrides, requires_approval
from src.escalation.ultrabert import FineTunedClassifier, balanced_training_rows, case_text, case_texts, class_weights

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/processed/escalation"
MODELS = OUT / "ultrabert"
REPORT = ROOT / "reports/step_07"
FINE, ZERO = "ultrabert_fine_tuned", "ultrabert_zero_shot"


def case(**changes):
    base = {"on_hand": 50, "on_order": 0, "fc_mean_14": 14.0, "fc_q90_sum_14": 25.0, "cover_days": 50.0, "days_to_next_due": np.nan, "units_due_14": 0.0, "units_overdue": 0.0, "days_overdue_max": 0.0, "open_pos": 0.0,
            "ordered_today": 0, "supplier_late_rate": 0.1, "supplier_recent_late_rate": 0.1, "stockout_risk": 0.02, "stockout_risk_lightgbm": 0.02, "value_band": "low"}
    return pd.DataFrame([{**base, **changes}])


@pytest.fixture(scope="module")
def test_cases():
    return pd.read_parquet(OUT / "classified_test.parquet")


@pytest.fixture(scope="module")
def choices():
    return pd.read_parquet(MODELS / "test_choices.parquet")


@pytest.fixture(scope="module")
def fine():
    return np.load(MODELS / "test_fine_tuned.npy")


@pytest.fixture(scope="module")
def zero():
    return np.load(MODELS / "test_zero_shot.npy")


def test_every_test_case_was_scored_by_both_routes(test_cases, fine, zero):
    assert fine.shape == (len(test_cases), 4)
    assert zero.shape == (len(test_cases), 4)
    assert np.isfinite(fine).all() and np.isfinite(zero).all()


def test_fine_tuned_probabilities_sum_to_one(fine):
    assert fine.sum(axis=1) == pytest.approx(np.ones(len(fine)), abs=1e-3)


def test_saved_choices_line_up_with_the_jev_cases(test_cases, choices):
    assert (choices.id.to_numpy() == test_cases.id.to_numpy()).all()
    assert (choices.origin.to_numpy() == test_cases.origin.to_numpy()).all()
    assert (choices.reference.to_numpy() == test_cases.reference.to_numpy()).all()


def test_choices_read_each_score_column_in_its_own_order(choices, fine, zero):
    assert (choices[FINE].to_numpy() == np.array(CATEGORIES, dtype=object)[fine.argmax(axis=1)]).all()
    assert (choices[ZERO].to_numpy() == np.array(list(CRITERIA), dtype=object)[zero.argmax(axis=1)]).all()


def test_saved_choices_are_valid_categories(choices):
    for column in [FINE, f"{FINE}_override", ZERO, f"{ZERO}_override"]:
        assert set(choices[column]) <= set(CATEGORIES), column


def test_reported_metrics_can_be_recomputed(choices):
    table = pd.read_csv(REPORT / "metrics_test.csv").set_index("model")
    for name in [FINE, ZERO]:
        truth, picked = choices.reference.to_numpy(dtype=object), choices[name].to_numpy(dtype=object)
        assert table.loc[name, "macro_f1"] == pytest.approx(f1_score(truth, picked, labels=CATEGORIES, average="macro", zero_division=0), abs=1e-4)
        assert table.loc[name, "critical_recall"] == pytest.approx((picked[truth == "human_review"] == "human_review").mean(), abs=1e-4)
        assert table.loc[name, "false_alarm_share"] == pytest.approx((picked[truth == "monitor"] != "monitor").mean(), abs=1e-4)


def test_earlier_jev_and_policy_results_are_unchanged():
    table = pd.read_csv(REPORT / "metrics_test.csv").set_index("model")
    assert table.loc["jev", "macro_f1"] == 0.42845
    assert table.loc["jev", "critical_recall"] == 0.23551
    assert table.loc["policy_baseline", "macro_f1"] == 0.49139
    assert table.loc["policy_baseline", "critical_recall"] == 0.38406
    saved = json.loads((REPORT / "comparison.json").read_text())
    assert saved["macro_f1_jev_minus_policy"] == [-0.0851, -0.0628, -0.0384]


def test_all_routes_appear_in_the_results():
    assert {"jev", "policy_baseline", FINE, ZERO} <= set(pd.read_csv(REPORT / "metrics_test.csv").model)
    scenario = pd.read_csv(REPORT / "metrics_by_scenario.csv")
    assert scenario.groupby("scenario").model.nunique().eq(4).all()
    edge = pd.read_csv(REPORT / "edge_cases.csv")
    assert len(edge) == 10
    assert {"jev", "policy", FINE, ZERO} <= set(edge.columns)


def test_training_stayed_before_the_development_period_and_test():
    log = (REPORT / "ultrabert_run.log").read_text(encoding="utf-8", errors="replace")
    last_train = pd.Timestamp(re.search(r"last origin (\d{4}-\d{2}-\d{2})", log).group(1))
    first_dev = pd.Timestamp(re.search(r"first origin (\d{4}-\d{2}-\d{2})", log).group(1))
    assert last_train + pd.Timedelta(days=14) < first_dev


def test_test_cases_start_after_all_training_labels_end(test_cases):
    assert test_cases.origin.min() > pd.Timestamp("2015-09-14") + pd.Timedelta(days=14)
    assert test_cases.origin.min() >= pd.Timestamp("2016-01-01")


def test_best_epoch_was_chosen_on_development_data():
    state = json.loads((MODELS / "training_state.json").read_text())
    scores = [epoch["dev_macro_f1"] for epoch in state["epochs"]]
    assert state["done"] is True
    assert state["best_macro_f1"] == max(scores)
    assert state["best_epoch"] == int(np.argmax(scores))


@pytest.mark.parametrize("hidden", [{"reference": "human_review"}, {"future_lost_margin": 500.0}, {"label": 1}, {"scenario": "lean_policy"}, {"id": "ITEM_1"}, {"origin": pd.Timestamp("2016-01-04")}])
def test_case_text_ignores_outcomes_and_identifiers(hidden):
    assert case_text(next(case(**hidden).itertuples(index=False))) == case_text(next(case().itertuples(index=False)))


def test_case_text_carries_the_facts_jev_saw():
    text = case_text(next(case(on_hand=7, on_order=30, open_pos=2.0, stockout_risk=0.55, stockout_risk_lightgbm=0.61, value_band="high", cover_days=3.5, days_to_next_due=9.0).itertuples(index=False)))
    for piece in ["On hand 7", "3.5 days cover", "On order 30 in 2 orders", "55%", "61%", "Value high", "runs out before delivery"]:
        assert piece in text


def test_missing_delivery_date_is_written_as_unknown():
    assert "no delivery date due" in case_text(next(case().itertuples(index=False)))
    assert "Next delivery unknown" in case_text(next(case().itertuples(index=False)))


def test_class_weights_favor_rare_categories_and_are_limited():
    reference = ["monitor"] * 1000 + ["expedite_recommendation"] * 100 + ["reorder_recommendation"] * 10 + ["human_review"] * 1
    weights = dict(zip(CATEGORIES, class_weights(reference)))
    assert weights["monitor"] < weights["expedite_recommendation"] < weights["reorder_recommendation"]
    assert max(weights.values()) == 10.0


def test_balanced_rows_keep_all_stockouts():
    frame = pd.DataFrame({"reference": ["monitor"] * 100 + ["human_review"] * 5 + ["reorder_recommendation"] * 5, "row": range(110)})
    rows = balanced_training_rows(frame, 1, 42)
    assert (rows.reference != "monitor").sum() == 10
    assert (rows.reference == "monitor").sum() == 10


def test_predict_returns_scores_in_the_original_order():
    class Encoded(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return Encoded(input_ids=torch.tensor([[len(t)] for t in texts]))

    class Output:
        def __init__(self, logits):
            self.logits = logits

    class Model:
        def eval(self):
            return self

        def __call__(self, input_ids):
            return Output(torch.nn.functional.one_hot(input_ids[:, 0] % 4, 4).float() * 10)

    stand_in = FineTunedClassifier.__new__(FineTunedClassifier)
    stand_in.batch_size, stand_in.tokenizer, stand_in.model = 2, Tokenizer(), Model()
    stand_in._encode = lambda texts: stand_in.tokenizer(texts)
    frame = pd.concat([case(on_hand=v, on_order=w) for v, w in [(1000, 1000), (1, 0), (100, 10), (10, 100), (5, 5000)]], ignore_index=True)
    expected = np.array([len(t) % 4 for t in case_texts(frame)])
    assert (stand_in.predict(frame).argmax(axis=1) == expected).all()


def test_overrides_never_change_a_valid_fine_tuned_choice(choices, test_cases):
    changed, _ = apply_overrides(test_cases, choices[FINE])
    assert (changed == choices[f"{FINE}_override"].to_numpy(dtype=object)).all()


def test_ultrabert_choices_still_need_a_person_to_approve(choices):
    needed = requires_approval(choices[FINE])
    assert (needed == (choices[FINE] != "monitor").to_numpy()).all()
    assert needed.any()


def test_sending_to_jev_is_still_off():
    assert jev.SENDING_ENABLED is False
    with pytest.raises(RuntimeError):
        jev.send({}, "not a key")


@pytest.mark.parametrize("path", ["src/escalation/ultrabert.py", "scripts/run_ultrabert.py", "scripts/compare_ultrabert.py"])
def test_ultrabert_code_sends_nothing_to_jev(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    for term in ["classify", "send(", "TYPESAFE_API_KEY", "api.typesafe", "requests.", "httpx"]:
        assert term not in source, term
