"""Run one review: the model gathers evidence through tools, then submits a recommendation that code checks."""
import ast
import json
import re
import time

from pydantic import ValidationError

from src.agent.review import review
from src.agent.tools import REQUIRED_TOOLS, Recommendation, call_tool, tool_schemas

MAX_TURNS = 8
TOOL_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
SYSTEM = """You review one item's inventory for the next 14 days and recommend one category for a person to approve. You cannot place orders.
Categories:
- monitor: stock looks sufficient, no action needed.
- reorder_recommendation: a stockout looks likely and no order is on the way.
- expedite_recommendation: a stockout looks likely and an open order may arrive too late.
- human_review: a stockout looks likely and would be costly, or the evidence conflicts.
Call get_inventory_status, get_demand_forecast, get_stockout_risk, get_open_orders and get_escalation_suggestion, all at once if you can. Use search_policy if a rule is unclear.
Weigh the evidence yourself; the escalation suggestion is one input and can be wrong. Then call submit_recommendation. Cite only values from tool results, and say what is unknown.
Reply only with tool calls, with no other text."""


def _read_call(chunk):
    """Read one call written as JSON or as a Python dictionary, without running anything."""
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return ast.literal_eval(chunk)


def parse_tool_calls(text):
    """Return well-formed calls, a count of malformed ones, and a count of calls written without the required tags."""
    tagged = TOOL_CALL.findall(text)
    chunks = tagged or [line for line in text.splitlines() if line.strip().startswith("{")]
    calls, malformed = [], 0
    for chunk in chunks:
        try:
            item = _read_call(chunk.strip())
            calls.append({"name": str(item["name"]), "arguments": item.get("arguments", {})})
        except (ValueError, SyntaxError, KeyError, TypeError, AttributeError):
            malformed += 1
    return calls, malformed, 0 if tagged else len(calls)


def _check_submission(arguments, seen):
    """Accept a recommendation only after the required evidence came back in an earlier turn."""
    missing = sorted(REQUIRED_TOOLS - seen)
    if missing:
        return {"ok": False, "error": "evidence missing, call these first and read the results: " + ", ".join(missing)}
    try:
        return {"ok": True, "recommendation": Recommendation(**(arguments if isinstance(arguments, dict) else {})).model_dump(), "result": {"received": True}}
    except ValidationError as error:
        return {"ok": False, "error": "invalid recommendation: " + "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors())}


def _finish(trace, submission):
    """Apply the checks in code. Without a valid submission the case goes to a person, and the model's answer stays empty."""
    risk = trace["evidence"].get("get_stockout_risk", {})
    orders = trace["evidence"].get("get_inventory_status", {}).get("open_orders")
    if submission is None:
        trace["recommendation"] = None
        trace["review"] = {"final_category": "human_review", "approval_required": True, "order_placed": False, "flags": ["agent_failed:no valid recommendation"]}
        return trace
    trace["recommendation"] = submission
    trace["review"] = review(submission["category"], submission["explanation"], trace["evidence"], [risk.get("tabpfn_risk_percent"), risk.get("lightgbm_risk_percent")], orders)
    return trace


def run_case(llm, box, case_id, schemas=None, max_turns=MAX_TURNS):
    """Review one case and return a full trace. A run that does not finish goes to a person, flagged as failed."""
    schemas = schemas or tool_schemas()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Review case_id {case_id}."}]
    trace = {"case_id": case_id, "turns": [], "tool_calls": [], "evidence": {}, "input_tokens": 0, "output_tokens": 0, "malformed_calls": 0, "untagged_calls": 0, "invalid_submissions": 0, "status": "failed"}
    started = time.time()
    submission = None
    for turn in range(max_turns):
        output = llm.generate(messages, schemas)
        trace["input_tokens"] += output["input_tokens"]
        trace["output_tokens"] += output["output_tokens"]
        trace["turns"].append({"turn": turn, "model_text": output["text"], "input_tokens": output["input_tokens"], "output_tokens": output["output_tokens"], "seconds": round(output["seconds"], 2)})
        messages.append({"role": "assistant", "content": output["text"]})
        calls, malformed, untagged = parse_tool_calls(output["text"])
        trace["malformed_calls"] += malformed
        trace["untagged_calls"] += untagged
        if turn and output["text"] == trace["turns"][-2]["model_text"]:
            trace["stopped_repeating"] = True
            break
        if not calls:
            messages.append({"role": "user", "content": "Reply only with tool calls, each inside <tool_call></tool_call> tags. Finish by calling submit_recommendation."})
            continue
        seen_before_turn = set(trace["evidence"])
        for call in calls:
            if call["name"] == "submit_recommendation":
                result = _check_submission(call["arguments"], seen_before_turn)
                if result["ok"]:
                    submission = result["recommendation"]
                else:
                    trace["invalid_submissions"] += 1
            else:
                result = call_tool(box, call["name"], call["arguments"])
                if result["ok"]:
                    trace["evidence"][call["name"]] = result["result"]
            trace["tool_calls"].append({"turn": turn, **call, "ok": result["ok"], "output": result.get("result", result.get("error"))})
            messages.append({"role": "tool", "content": json.dumps(result.get("result", {"error": result.get("error")}))})
        if submission:
            trace["status"] = "submitted"
            break
    trace["seconds"] = round(time.time() - started, 2)
    return _finish(trace, submission)
