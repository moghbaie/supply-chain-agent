"""Step 11: check an agent explanation claim by claim against the tool results its run saw.

Code checks numbers, open-order claims and value bands. A local LLM checks qualitative claims, one at a time,
with the evidence and the project's definitions, and never sees the agent's category. Nothing here calls an API.
"""
import json
import random
import re
from dataclasses import asdict, dataclass, field

from src.agent.audit import DELAYED, NO_ORDERS, ORDER_CLAIM
from src.agent.review import NUMBER

TOOL_ORDER = ["get_inventory_status", "get_demand_forecast", "get_stockout_risk", "get_open_orders", "get_escalation_suggestion"]
ERROR_VERDICTS = {"wrong_value", "misattributed", "unsupported", "contradicted"}
WHOLE_TOLERANCE, DECIMAL_TOLERANCE = 1.0, 0.051

# Horizons such as "next 14 days" set the time window; they are not claims about a tool result.
HORIZON = re.compile(r"(?:next|last)[\s-]+(?:the[\s-]+)?(?:14|28|30)[\s-]+days?|over\s+(?:14|28)\s+days?", re.I)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z'\"(])")

# Each stated number takes its unit from the words right after it, so "7 units" cannot bind to days of cover.
UNIT_AFTER = [(re.compile(r"\s*%"), "percent"), (re.compile(r"\s*points?\b", re.I), "percent"), (re.compile(r"(?:\s+|-)days?\b", re.I), "days"), (re.compile(r"\s+units?\b", re.I), "units")]

# Field groups. Each value is a pattern over field names, and the unit kinds the group accepts.
GROUPS = {
    "on_hand": (r"inventory\.on_hand_units", {"units"}),
    "on_order": (r"inventory\.on_order_units", {"units"}),
    "cover": (r"inventory\.days_of_cover", {"days"}),
    "order_count": (r"inventory\.open_orders|orders\.count", {"count"}),
    "sales_rate": (r"inventory\.sales_per_day_last_28_days", {"rate"}),
    "sold_out": (r"inventory\.sold_out_days_last_28_days", {"days"}),
    "forecast": (r"forecast\.forecast_units_next_14_days", {"units"}),
    "low_estimate": (r"forecast\.low_estimate_units", {"units"}),
    "high_estimate": (r"forecast\.high_estimate_units", {"units"}),
    "stockout": (r"risk\.(?:tabpfn|lightgbm)_risk_percent", {"percent"}),
    "tabpfn": (r"risk\.tabpfn_risk_percent", {"percent"}),
    "lightgbm": (r"risk\.lightgbm_risk_percent", {"percent"}),
    "gap": (r"risk\.models_differ_by_points", {"percent"}),
    "order_units": (r"orders\.open_orders\[\d+\]\.units|orders\.total_units|orders\.units_due_next_14_days|orders\.units_overdue", {"units"}),
    "order_days": (r"orders\.open_orders\[\d+\]\.days_until_promised|orders\.days_to_next_delivery|orders\.days_overdue", {"days"}),
    "delay_risk": (r"orders\.open_orders\[\d+\]\.delay_risk_percent", {"percent"}),
    "late_rate": (r"orders\.supplier_late_percent_(?:long_run|last_30_days)", {"percent"}),
    "late_long": (r"orders\.supplier_late_percent_long_run", {"percent"}),
    "late_recent": (r"orders\.supplier_late_percent_last_30_days", {"percent"}),
    "probability": (r"suggestion\.probabilities_percent\.\w+", {"percent"}),
    "prob_monitor": (r"suggestion\.probabilities_percent\.monitor", {"percent"}),
    "prob_reorder": (r"suggestion\.probabilities_percent\.reorder_recommendation", {"percent"}),
    "prob_expedite": (r"suggestion\.probabilities_percent\.expedite_recommendation", {"percent"}),
    "prob_human": (r"suggestion\.probabilities_percent\.human_review", {"percent"}),
    "value_band": (None, {"percent", "days", "units", "count", "rate"}),  # a band has no number: a number bound here stays unbound
}

# Words that name a field. Earlier patterns win where two overlap, so "probability of stockout" is not read as a classifier probability.
CUES = [(re.compile(pattern, re.I), group) for pattern, group in [
    (r"days? of cover", "cover"),
    (r"days? until (?:the )?promised(?: date)?", "order_days"),
    (r"(?:next )?delivery (?:is )?in|delivery window|due in|arrive in|arriving in", "order_days"),
    (r"sales per day", "sales_rate"),
    (r"sold[- ]out days", "sold_out"),
    (r"low estimate", "low_estimate"),
    (r"high estimate", "high_estimate"),
    (r"value band (?:of|\()", "value_band"),  # only a number written straight after the band, such as "value band of 18 units"
    (r"delay risk|risk of delay", "delay_risk"),
    (r"late rates?|late percent", "late_rate"),
    (r"differ(?:s|ing)? by|difference", "gap"),
    (r"(?:probability|chance|risk) of (?:a )?stockout|stockout (?:risk|probability|chance)", "stockout"),
    (r"probability of expedit\w*", "prob_expedite"),
    (r"probability of reorder\w*", "prob_reorder"),
    (r"probability", "probability"),
    (r"units on hand|on[- ]hand(?: units)?", "on_hand"),
    (r"units on order|on[- ]order(?: units)?", "on_order"),
    (r"units?(?= due\b)|units? overdue|total of", "order_units"),  # "units" alone, so "due in 9 days" stays a cue for the days
    (r"open orders?|orders? pending|pending orders?", "order_count"),
    (r"demand forecast|forecast(?:ed)?(?: demand)?|demand", "forecast"),
    (r"current inventory|inventory|stock(?: level)? is currently|current stock", "on_hand"),
    (r"\b(?:chance|risk)\b", "stockout"),
]]

# Words that narrow a group once it is chosen, such as which model a stockout risk came from.
QUALIFIERS = {
    "stockout": [(re.compile(r"tabpfn", re.I), "tabpfn"), (re.compile(r"lightgbm", re.I), "lightgbm")],
    "late_rate": [(re.compile(r"long[- ]run", re.I), "late_long"), (re.compile(r"last[- ]30[- ]days|recent", re.I), "late_recent")],
    "probability": [(re.compile(r"monitor", re.I), "prob_monitor"), (re.compile(r"reorder", re.I), "prob_reorder"), (re.compile(r"expedit", re.I), "prob_expedite"),
                    (re.compile(r"human[ _]review", re.I), "prob_human")],
}
BOTH_MODELS = re.compile(r"\bboth models\b", re.I)
RESPECTIVELY = re.compile(r"\brespectively\b", re.I)
SEPARATOR = re.compile(r",|;|\band\b|\bwhile\b", re.I)
ATTACHED_AFTER = 25  # characters allowed between a number and a qualifier written after it, as in "5.3% for the last 30 days"

ORDER_EXISTS = re.compile("|".join([ORDER_CLAIM.pattern, r"there (?:is|are) (?:an? |\d+ )?open orders?", r"\bopen orders? (?:of|has|have|show|shows)\b", r"\b\d+ open orders?",
                                    r"\bopen orders? (?:is|are) (?:also|\d)", r"\ban open order\b", r"\b\d+ orders? pending"]), re.I)
BAND_CLAIM = re.compile(r"(?<!below the )(?<!above the )(?<!within the )\b(low|medium|high) value band\b", re.I)

# Qualitative claims go to the LLM. Phrases checked elsewhere, or that describe the agent's own uncertainty, are removed first.
NOT_QUALITATIVE = re.compile(r"low estimate|high estimate|late rates?|delay risk|(?:(?:low|medium|high) )?value band|(?:high|low|medium) (?:in )?uncertainty(?: level)?|uncertainty is \w+|medium unsettled|"
                             r"high (?:probability|confidence)(?= (?:of )?\(?\d)|with a high probability(?= of \d)", re.I)
QUALITATIVE = re.compile(r"\b(?:high|higher|low|lower|medium|moderate|likely|unlikely|sufficient|insufficient|enough|concern|exceed\w*|below|above|within|over the current|"
                         r"on track|in time|too late|late|delay|delays|delayed|approaching|urgen\w*|need|needed|conflicts?|strongly|warrant|not zero|relatively|empty|zero)\b", re.I)

DEFINITIONS = """Definitions used in this project (project choices, not measured thresholds):
- Stockout risk is the chance of a stockout in the next 14 days, given by two models, TabPFN and LightGBM, in percent.
- High or likely stockout risk: at least one model at 50% or more. Low stockout risk: both models under 20%. Between 20% and 50% is neither clearly high nor clearly low.
- Supplier late rates under 10% are low. 10% or more is not low.
- An open order is delayed only if it has units overdue. Delay risk is the chance an open order arrives after its promised date.
- Stock covers the forecast when on-hand units are at least the 14-day forecast."""

JUDGE_SYSTEM = f"""You check one claim from an explanation of an inventory review against the tool results that review saw.
Use only the tool results and the definitions. Do not use outside knowledge.
{DEFINITIONS}
Answer "supported" if the tool results show the claim is true, "contradicted" if they show it is false, and "unclear" if they cannot settle it.
Reply with JSON only, in this form: {{"verdict": "supported" or "contradicted" or "unclear", "reason": "one sentence"}}"""
LLM_VERDICTS = {"supported", "contradicted", "unclear"}


@dataclass
class Claim:
    case_id: str
    sentence: int
    clause_index: int
    clause: str
    kind: str  # number, open_orders, value_band, qualitative
    text: str
    sentence_text: str = ""
    stated: float | None = None
    unit: str | None = None
    group: str | None = None
    require_all: bool = False
    fields: list = field(default_factory=list)
    actual: list = field(default_factory=list)
    verdict: str = ""
    detail: str = ""
    position: int = 0  # order within the explanation, so each claim ID is unique

    @property
    def claim_id(self):
        return f"{self.case_id}-{self.position:02d}"

    def row(self):
        return {"claim_id": self.claim_id, **{k: v for k, v in asdict(self).items() if k not in ("sentence_text", "position")},
                "fields": ";".join(self.fields), "actual": ";".join(map(str, self.actual))}


# Splitting ------------------------------------------------------------------------------------------------------------

def sentences(text):
    return [part.strip() for part in SENTENCE_END.split(text.strip()) if part.strip()]


def clauses(sentence):
    """Split a sentence at commas and semicolons outside parentheses."""
    parts, depth, start = [], 0, 0
    for position, char in enumerate(sentence):
        depth += (char == "(") - (char == ")")
        if depth == 0 and char in ",;" and sentence[position + 1:position + 2] == " " and not sentence[position + 2:].lower().startswith("respectively"):
            parts.append(sentence[start:position])
            start = position + 1
    parts.append(sentence[start:])
    return [part.strip().rstrip(".").strip() for part in parts if part.strip().rstrip(".").strip()]


def stated_numbers(clause):
    """Return (text, value, start, end, unit) for each number that is not part of a horizon."""
    horizons = [match.span() for match in HORIZON.finditer(clause)]
    found = []
    for match in NUMBER.finditer(clause):
        start, end = match.span()
        if any(a <= start and end <= b for a, b in horizons):
            continue
        rest = clause[end:]
        unit = next((kind for pattern, kind in UNIT_AFTER if pattern.match(rest)), None)
        found.append((match.group(), float(match.group().replace(",", "")), start, end, unit))
    return found


def cue_spans(clause):
    """Find field cues, keeping the earlier pattern where two overlap."""
    kept, text = [], clause.replace("_", " ")  # raw field names such as delay_risk_percent count as words
    for pattern, group in CUES:
        for match in pattern.finditer(text):
            if not any(match.start() < b and a < match.end() for a, b, _ in kept):
                kept.append((match.start(), match.end(), group))
    return sorted(kept)


def compatible(group, unit):
    if group == "order_count":
        return unit in (None, "units", "days", "percent")
    return unit is None or unit in GROUPS[group][1]


def resolve_count_cue(group, unit):
    """"open order" names a count, but "open order of 18 units" names units and "open order due in 5 days" names days."""
    if group != "order_count":
        return group
    return {"units": "order_units", "days": "order_days", "percent": "delay_risk"}.get(unit, group)


def nearest(spans, start, end, clause=None):
    """Pick the span closest to [start, end), ranking spans across a comma last; ties go to the span before."""
    best = None
    for a, b, *rest in spans:
        if a < end and start < b:
            continue
        distance = start - b if b <= start else a - end
        between = (clause[b:start] if b <= start else clause[end:a]) if clause else ""
        key = ("," in between or ";" in between, distance, 0 if b <= start else 1)
        if best is None or key < best[0]:
            best = (key, (a, b, *rest))
    return best[1] if best else None


def qualify(group, clause, numbers, index):
    """Narrow a group with a qualifier. In order: a "respectively" list, a qualifier attached after the number ("5.3% last 30 days"),
    one attached before it ("tabpfn risk of 2.2%"), then the nearest in the clause. Attached means no comma, "and" or "while" between."""
    options = QUALIFIERS.get(group)
    if not options:
        return group, False
    text = clause.replace("_", " ")
    marks = sorted((m.start(), m.end(), refined) for pattern, refined in options for m in pattern.finditer(text))
    start, end = numbers[index][2], numbers[index][3]
    if RESPECTIVELY.search(clause):
        same = [n for n in numbers if n[5] == group]
        position = [n[2] for n in same].index(start)
        after = [m for m in marks if m[0] >= same[-1][3]]
        if len(after) == len(same):
            return after[position][2], False
    next_start = numbers[index + 1][2] if index + 1 < len(numbers) else len(clause)
    previous_end = numbers[index - 1][3] if index > 0 else 0
    after = [m for m in marks if end <= m[0] and m[1] <= next_start and m[0] - end <= ATTACHED_AFTER and not SEPARATOR.search(text[end:m[0]])]
    if after:
        return after[0][2], False
    before = [m for m in marks if previous_end <= m[0] and m[1] <= start and not SEPARATOR.search(text[m[1]:start])]
    if before:
        return before[-1][2], False
    if marks:
        return nearest(marks, start, end)[2], False
    return group, group == "stockout" and bool(BOTH_MODELS.search(clause))


def bind_numbers(clause):
    """Return (text, value, unit, group or None, require_all) for each stated number in a clause."""
    numbers = stated_numbers(clause)
    cues = cue_spans(clause)
    chosen = []
    for text, value, start, end, unit in numbers:
        usable = [c for c in cues if compatible(c[2], unit)]
        cue = nearest(usable, start, end, clause)
        chosen.append((text, value, start, end, unit, resolve_count_cue(cue[2], unit) if cue else None))
    bound = []
    for index, (text, value, start, end, unit, group) in enumerate(chosen):
        require_all = False
        if group is not None:
            group, require_all = qualify(group, clause, chosen, index)
        if group == "value_band":
            group = None
        bound.append((text, value, unit, group, require_all))
    return bound


def extract_claims(case_id, explanation):
    """Split an explanation into claims. Needs no evidence, so coverage can be measured on any explanation."""
    claims = []
    for s_index, sentence in enumerate(sentences(explanation)):
        for c_index, clause in enumerate(clauses(sentence)):
            base = dict(case_id=case_id, sentence=s_index, clause_index=c_index, clause=clause, sentence_text=sentence)
            for text, value, unit, group, require_all in bind_numbers(clause):
                claims.append(Claim(**base, kind="number", text=text, stated=value, unit=unit, group=group, require_all=require_all))
            order_text = ""
            for pattern, label in [(NO_ORDERS, "no open orders"), (DELAYED, "open orders are delayed"), (ORDER_EXISTS, "an open order exists")]:
                match = pattern.search(clause)
                if match:
                    claims.append(Claim(**base, kind="open_orders", text=label))
                    order_text = match.group()
                    break
            band = BAND_CLAIM.search(clause)
            if band:
                claims.append(Claim(**base, kind="value_band", text=band.group(1).lower()))
            rest = NOT_QUALITATIVE.sub(" ", clause.replace(order_text, " ") if order_text else clause)
            if QUALITATIVE.search(rest):
                claims.append(Claim(**base, kind="qualitative", text=clause))
    for position, claim in enumerate(claims):
        claim.position = position
    return claims


# Code verdicts --------------------------------------------------------------------------------------------------------

def flatten(evidence):
    """Name every numeric field in the evidence, plus the order count and total ordered units."""
    names = {"get_inventory_status": "inventory", "get_demand_forecast": "forecast", "get_stockout_risk": "risk", "get_open_orders": "orders", "get_escalation_suggestion": "suggestion"}
    flat = {}

    def walk(prefix, value):
        if isinstance(value, bool):
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix}.{key}", item)
        elif isinstance(value, list):
            for position, item in enumerate(value):
                walk(f"{prefix}[{position}]", item)
        elif value is None or isinstance(value, (int, float)):
            flat[prefix] = None if value is None else float(value)

    for tool, short in names.items():
        if tool in evidence:
            walk(short, {k: v for k, v in evidence[tool].items() if k != "missing_fields"})
    listed = evidence.get("get_open_orders", {}).get("open_orders")
    if isinstance(listed, list):
        flat["orders.count"] = float(len(listed))
        flat["orders.total_units"] = float(sum(item.get("units") or 0 for item in listed))
    return flat


def matches(stated_text, stated, actual):
    """A whole number may round or truncate the actual value. A decimal must match to its last digit."""
    if actual is None:
        return False
    tolerance = DECIMAL_TOLERANCE if "." in stated_text else WHOLE_TOLERANCE
    return abs(stated - actual) < tolerance if tolerance == WHOLE_TOLERANCE else abs(stated - actual) <= tolerance


def check_number(claim, flat):
    """Compare a stated number with the field it names. Only a non-zero decimal can be misattributed: a whole number, or zero, can equal another field by chance."""
    elsewhere = [name for name, value in flat.items() if matches(claim.text, claim.stated, value)]
    if claim.group is None:
        claim.verdict = "unverified" if elsewhere else "unsupported"
        claim.detail = ("in " + ",".join(elsewhere)) if elsewhere else "in no tool result"
        return claim
    pattern = re.compile(GROUPS[claim.group][0])
    named = [name for name in flat if pattern.fullmatch(name)]
    claim.fields = named
    claim.actual = [flat[name] for name in named]
    present = [name for name in named if flat[name] is not None]
    if not present:
        claim.verdict, claim.detail = "unsupported", "named field is missing from the tool results"
        return claim
    hits = [name for name in present if matches(claim.text, claim.stated, flat[name])]
    if (claim.require_all and len(hits) == len(present)) or (not claim.require_all and hits):
        claim.verdict = "supported"
        return claim
    others = [name for name in elsewhere if name not in named]
    if others and "." in claim.text and claim.stated != 0:
        claim.verdict, claim.detail = "misattributed", "value of " + ",".join(others)
    else:
        claim.verdict = "wrong_value"
        claim.detail = "both models must match" if claim.require_all and hits else ""
    return claim


def check_orders(claim, evidence):
    orders = evidence.get("get_open_orders")
    if orders is None:
        claim.verdict, claim.detail = "unsupported", "open orders tool result missing"
        return claim
    listed = orders.get("open_orders") or []
    if claim.text == "no open orders":
        truth = not listed
    elif claim.text == "open orders are delayed":
        truth = bool(listed) and bool(orders.get("units_overdue"))
    else:
        truth = bool(listed)
    claim.verdict = "supported" if truth else "contradicted"
    claim.detail = f"{len(listed)} listed, {orders.get('units_overdue')} units overdue"
    return claim


def check_band(claim, evidence):
    band = evidence.get("get_inventory_status", {}).get("value_band")
    claim.verdict = "unsupported" if band is None else ("supported" if band == claim.text else "contradicted")
    claim.detail = f"value band is {band}"
    return claim


# LLM verdicts ---------------------------------------------------------------------------------------------------------

def tool_orders(count, seed=11):
    """The fixed order first, then distinct shuffled orders, so position bias can be measured."""
    found, generator = [list(TOOL_ORDER)], random.Random(seed)
    while len(found) < count:
        order = list(TOOL_ORDER)
        generator.shuffle(order)
        if order not in found:
            found.append(order)
    return found


def judge_prompt(claim, evidence, order=TOOL_ORDER):
    """Messages for one claim. The prompt holds the claim, its sentence and the tool results; never the agent's category."""
    results = "\n".join(f"{tool}: {json.dumps(evidence[tool])}" for tool in order if tool in evidence)
    user = f"Tool results:\n{results}\n\nSentence the claim comes from: \"{claim.sentence_text}\"\nClaim to check, only this part: \"{claim.text}\""
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_reply(text):
    """Accept a JSON object, optionally in a code fence, and nothing else. Return (verdict, reason) or None."""
    body = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.S)
    if fence:
        body = fence.group(1)
    try:
        reply = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(reply, dict) or reply.get("verdict") not in LLM_VERDICTS:
        return None
    return reply["verdict"], str(reply.get("reason", ""))


def judge_qualitative(claim, evidence, llm, orders=(TOOL_ORDER,)):
    """Ask the LLM once per tool order. The claim's verdict is the one from the fixed order; changes across orders are recorded."""
    verdicts = []
    for order in orders:
        reply = llm.generate(judge_prompt(claim, evidence, order), tools=None)
        parsed = parse_reply(reply["text"])
        verdicts.append("judge_error" if parsed is None else parsed[0])
        if len(verdicts) == 1:
            claim.detail = reply["text"].strip()[:300] if parsed is None else parsed[1]
    claim.verdict = verdicts[0]
    claim.actual = verdicts
    return claim


def judge_explanation(case_id, explanation, evidence, llm=None, orders=1):
    """Return every claim in one explanation with a verdict. Without an LLM, qualitative claims are not_checked."""
    flat = flatten(evidence)
    order_list = tool_orders(orders)
    claims = extract_claims(case_id, explanation)
    for claim in claims:
        if claim.kind == "number":
            check_number(claim, flat)
        elif claim.kind == "open_orders":
            check_orders(claim, evidence)
        elif claim.kind == "value_band":
            check_band(claim, evidence)
        elif llm is None:
            claim.verdict = "not_checked"
        else:
            judge_qualitative(claim, evidence, llm, order_list)
    return claims


def summarize(case_id, claims):
    errors = [c for c in claims if c.verdict in ERROR_VERDICTS]
    return {"case_id": case_id, "claims": len(claims), "errors": len(errors), "has_error": bool(errors),
            "code_error": any(c.kind != "qualitative" for c in errors), "error_claims": " | ".join(f"{c.verdict}: {c.text} ({c.group or c.kind})" for c in errors)}
