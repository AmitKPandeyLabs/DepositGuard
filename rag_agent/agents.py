import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

import chromadb
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from tools import get_account_profile, get_regulatory_context, get_peer_comparison, make_escalation_decision

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
ESCALATION_DIR = os.path.join(BASE_DIR, "outputs", "escalation_reports")

# If the Investigation node's confidence score falls below this threshold, the conditional
# edge routes back to Investigation for another (broadened) retrieval pass instead of
# proceeding straight to Escalation. Capped by MAX_RETRIEVAL_PASSES to guarantee termination.
CONFIDENCE_THRESHOLD = 0.80
MAX_RETRIEVAL_PASSES = 2

RECOMMENDED_ACTION_BY_TIER = {
    "HIGH": "Immediate account review — consider temporary freeze pending investigation",
    "MEDIUM": "Queue for analyst review within 24 hours",
    "LOW": "Passive monitoring — no action required",
}

ANALYST_QUESTION_BY_TOP_FEATURE = {
    "device_os_windows": "Can you confirm the device used during account opening matches the applicant's stated location?",
    "proposed_credit_limit": "Does the requested credit limit align with the applicant's verified income?",
}
DEFAULT_ANALYST_QUESTION = "Please verify the identity documents submitted during account opening"

FIXED_CRITIQUE = "Assessment based on behavioral signals only — identity verification documents not reviewed"


class FraudCaseState(TypedDict, total=False):
    """Shared state carried across the Triage -> Investigation -> Escalation graph."""

    account_id: int

    # Triage
    profile: Dict[str, Any]
    risk_tier: str
    fraud_probability: float
    routing_decision: str
    triage_reason: str

    # Investigation
    regulatory_context: Dict[str, Any]
    peer_comparison: Dict[str, Any]
    initial_assessment: str
    critique: str
    refined_assessment: str
    root_cause: str
    confidence: str
    confidence_score: float
    regulatory_flags: List[str]
    recommended_action: str
    analyst_question: str
    retrieval_pass: int

    # Escalation
    report: Dict[str, Any]


def _mock_confidence(fraud_probability):
    if fraud_probability > 0.90:
        return 0.92, "Very High (92%)"
    elif fraud_probability > 0.75:
        return 0.78, "High (78%)"
    elif fraud_probability > 0.50:
        return 0.61, "Moderate (61%)"
    return 0.35, "Low (35%)"


def _mock_root_cause(top_shap_features):
    top_three = top_shap_features[:3]
    signals = ", ".join(f"{f['feature']} ({f['shap_value']:+.2f})" for f in top_three)
    return f"Account flagged primarily due to: {signals}"


def _mock_recommended_action(risk_tier):
    return RECOMMENDED_ACTION_BY_TIER.get(risk_tier, RECOMMENDED_ACTION_BY_TIER["MEDIUM"])


def _mock_regulatory_flags(matches):
    combined_text = " ".join(m["text"] for m in matches)
    flags = []
    if re.search(r"\bACH\b", combined_text, re.IGNORECASE):
        flags.append("Reg E — unauthorized transfer investigation required")
    if re.search(r"\bSARs?\b", combined_text, re.IGNORECASE):
        flags.append("BSA — evaluate for SAR filing")
    if "new account" in combined_text.lower():
        flags.append("New Account Fraud — enhanced due diligence")
    return flags


def _mock_analyst_question(top_shap_features):
    if not top_shap_features:
        return DEFAULT_ANALYST_QUESTION
    top_feature = top_shap_features[0]["feature"]
    return ANALYST_QUESTION_BY_TOP_FEATURE.get(top_feature, DEFAULT_ANALYST_QUESTION)


def _mock_refined_assessment(root_cause, regulatory_flags):
    if regulatory_flags:
        return f"{root_cause}. Regulatory considerations: {'; '.join(regulatory_flags)}."
    return f"{root_cause}. No specific regulatory flags identified from available context."


def _regulatory_query_topic(profile, retrieval_pass):
    """First pass mirrors the original fixed query; retries broaden it around the top SHAP signal."""
    if retrieval_pass <= 1:
        return "new account fraud"
    top_shap_features = profile.get("top_shap_features") or []
    if top_shap_features:
        return f"new account fraud {top_shap_features[0]['feature']} unauthorized transaction"
    return "new account fraud unauthorized transaction"


# ---------------------------------------------------------------------------------------
# Investigation backend dispatch. USE_MOCK_INVESTIGATION is the single switch: flipping it
# to False routes the Investigation node through _investigate_live (Gemini primary, Groq
# fallback) instead of the deterministic mock — investigation_agent and the rest of the
# graph are unaffected either way. _investigate_live always returns a dict with the exact
# same keys/types as _investigate_mock (see its docstring), falling back to the mock itself
# on any API or parsing failure so the pipeline never crashes.
# ---------------------------------------------------------------------------------------
USE_MOCK_INVESTIGATION = False

_INVESTIGATION_JSON_INSTRUCTIONS = """Return ONLY a single valid JSON object — no markdown code fences, no prose before or after — with EXACTLY these five keys and types, and no other keys:

{
  "confidence": "<string — a human-readable confidence level with a percentage, e.g. 'High (78%)'>",
  "confidence_score": <float between 0.0 and 1.0, matching the percentage in "confidence">,
  "root_cause": "<string — one to two sentences explaining why this account was flagged>",
  "regulatory_flags": ["<string>", "..."],
  "critique": "<string — one to two sentence self-critique of this assessment's blind spots or limitations>"
}

Rules:
- root_cause must be grounded ONLY in the SHAP features listed below. Do not invent or reference any feature not listed.
- regulatory_flags must be grounded ONLY in the retrieved regulatory context provided below. Return an empty list ([]) if nothing in the provided context applies. Do not cite regulations not present in the context.
- critique should note a genuine limitation of this assessment (e.g., evidence not reviewed, signals not available), consistent with an analyst's self-review of a behavioral-signal-only model.
- confidence_score must be a plain float, not a string."""


def _build_investigation_prompt(profile, regulatory_context, peer_comparison):
    shap_lines = "\n".join(
        f"- {f['feature']}: {f['shap_value']:+.4f}" for f in profile["top_shap_features"]
    )
    reg_text = "\n\n".join(
        f"[Source: {m['source']}]\n{m['text']}" for m in regulatory_context["matches"]
    )

    return f"""You are a fraud-investigation analyst assistant at a bank. Analyze the following \
HIGH-risk deposit account flagged by an ML fraud model, reasoning ONLY from the evidence provided below.

ACCOUNT RISK PROFILE
- Risk tier: {profile['risk_tier']}
- Fraud probability: {profile['fraud_probability']:.4f}
- Peer percentile: {peer_comparison['percentile']} ({peer_comparison['interpretation']})

TOP SHAP FEATURES (the model's risk drivers for this account, signed contribution to fraud probability):
{shap_lines}

RETRIEVED REGULATORY / FRAUD-TYPOLOGY CONTEXT (query: "{regulatory_context['query']}"):
{reg_text}

{_INVESTIGATION_JSON_INSTRUCTIONS}"""


def _strip_json_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _parse_investigation_json(raw_text):
    """Parse and validate the LLM's JSON response against the five LLM-owned fields.
    Raises on anything that doesn't match — callers must catch and fall back to the mock.
    """
    data = json.loads(_strip_json_fences(raw_text))

    regulatory_flags = data["regulatory_flags"]
    if not isinstance(regulatory_flags, list):
        raise ValueError("regulatory_flags must be a list")

    return {
        "confidence": str(data["confidence"]),
        "confidence_score": float(data["confidence_score"]),
        "root_cause": str(data["root_cause"]),
        "regulatory_flags": [str(flag) for flag in regulatory_flags],
        "critique": str(data["critique"]),
    }


def _call_gemini(prompt):
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=GEMINI_API_KEY)
    return llm.invoke(prompt).content


def _call_groq(prompt):
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=GROQ_API_KEY)
    return llm.invoke(prompt).content


def _call_investigation_llm(prompt):
    """Gemini primary, Groq fallback. Returns raw response text, or None if both fail."""
    try:
        return _call_gemini(prompt)
    except Exception:
        pass

    try:
        return _call_groq(prompt)
    except Exception:
        return None


def _investigate_mock(profile, regulatory_context, peer_comparison):
    """Deterministic, rule-based investigation over real fraud_probability, real SHAP
    features, and real RAG-retrieved regulatory context — no LLM call.
    """
    fraud_probability = profile["fraud_probability"]
    risk_tier = profile["risk_tier"]
    top_shap_features = profile["top_shap_features"]

    confidence_score, confidence = _mock_confidence(fraud_probability)
    root_cause = _mock_root_cause(top_shap_features)
    recommended_action = _mock_recommended_action(risk_tier)
    regulatory_flags = _mock_regulatory_flags(regulatory_context["matches"])
    analyst_question = _mock_analyst_question(top_shap_features)

    initial_assessment = (
        f"{root_cause}. Confidence: {confidence}. {peer_comparison['interpretation']} "
        f"Recommended action: {recommended_action}"
    )

    # Self-critique (fixed, per the known blind spot of a behavioral-signal-only model)
    critique = FIXED_CRITIQUE

    # Refine, incorporating the critique's regulatory context
    refined_assessment = _mock_refined_assessment(root_cause, regulatory_flags)

    return {
        "confidence": confidence,
        "confidence_score": confidence_score,
        "root_cause": root_cause,
        "recommended_action": recommended_action,
        "regulatory_flags": regulatory_flags,
        "analyst_question": analyst_question,
        "initial_assessment": initial_assessment,
        "critique": critique,
        "refined_assessment": refined_assessment,
    }


def _investigate_live(profile, regulatory_context, peer_comparison):
    """Live investigation over real fraud_probability, real SHAP features, and real
    RAG-retrieved regulatory context. Gemini is tried first, Groq on any Gemini failure;
    the LLM only determines confidence, root_cause, regulatory_flags, and critique —
    recommended_action and analyst_question stay rule-based (same helpers as the mock),
    and initial_assessment/refined_assessment are composed the same way the mock composes
    them. On any API failure or JSON-parse/validation failure from both providers, falls
    back to _investigate_mock() so the pipeline never crashes. Returns a dict with the
    exact same keys/types as _investigate_mock().
    """
    risk_tier = profile["risk_tier"]
    top_shap_features = profile["top_shap_features"]

    prompt = _build_investigation_prompt(profile, regulatory_context, peer_comparison)
    raw_output = _call_investigation_llm(prompt)

    llm_fields = None
    if raw_output is not None:
        try:
            llm_fields = _parse_investigation_json(raw_output)
        except Exception:
            llm_fields = None

    if llm_fields is None:
        return _investigate_mock(profile, regulatory_context, peer_comparison)

    confidence = llm_fields["confidence"]
    confidence_score = llm_fields["confidence_score"]
    root_cause = llm_fields["root_cause"]
    regulatory_flags = llm_fields["regulatory_flags"]
    critique = llm_fields["critique"]

    recommended_action = _mock_recommended_action(risk_tier)
    analyst_question = _mock_analyst_question(top_shap_features)

    initial_assessment = (
        f"{root_cause}. Confidence: {confidence}. {peer_comparison['interpretation']} "
        f"Recommended action: {recommended_action}"
    )
    refined_assessment = _mock_refined_assessment(root_cause, regulatory_flags)

    return {
        "confidence": confidence,
        "confidence_score": confidence_score,
        "root_cause": root_cause,
        "recommended_action": recommended_action,
        "regulatory_flags": regulatory_flags,
        "analyst_question": analyst_question,
        "initial_assessment": initial_assessment,
        "critique": critique,
        "refined_assessment": refined_assessment,
    }


def triage_agent(state: FraudCaseState) -> Dict[str, Any]:
    """Triage node: routes an account based on its precomputed risk tier (HIGH -> investigate,
    MEDIUM -> flag, LOW -> auto-clear).
    """
    account_id = state["account_id"]
    profile = get_account_profile(account_id)

    if "error" in profile:
        return {
            "profile": profile,
            "routing_decision": "ERROR",
            "triage_reason": profile["error"],
        }

    risk_tier = profile["risk_tier"]
    fraud_probability = profile["fraud_probability"]

    if risk_tier == "HIGH":
        routing_decision = "INVESTIGATE"
        reason = f"HIGH risk tier (fraud probability {fraud_probability:.3f}) — routed to Investigation Agent."
    elif risk_tier == "MEDIUM":
        routing_decision = "FLAG_FOR_REVIEW"
        reason = f"MEDIUM risk tier (fraud probability {fraud_probability:.3f}) — flagged for manual analyst review."
    else:
        routing_decision = "AUTO_CLEAR"
        reason = f"LOW risk tier (fraud probability {fraud_probability:.3f}) — auto-cleared, no further action."

    return {
        "profile": profile,
        "risk_tier": risk_tier,
        "fraud_probability": fraud_probability,
        "routing_decision": routing_decision,
        "triage_reason": reason,
    }


def investigation_agent(state: FraudCaseState) -> Dict[str, Any]:
    """Investigation node: RAG retrieval + tool-use + critique-refine assessment of a
    HIGH-risk account. Backend is controlled by USE_MOCK_INVESTIGATION above.
    """
    profile = state["profile"]
    retrieval_pass = state.get("retrieval_pass", 0) + 1

    # RAG retrieval — broadened on retry passes triggered by the low-confidence conditional edge
    query_topic = _regulatory_query_topic(profile, retrieval_pass)
    regulatory_context = get_regulatory_context(query_topic)
    peer_comparison = get_peer_comparison(profile["fraud_probability"])

    assessment = (
        _investigate_mock(profile, regulatory_context, peer_comparison)
        if USE_MOCK_INVESTIGATION
        else _investigate_live(profile, regulatory_context, peer_comparison)
    )

    return {
        "regulatory_context": regulatory_context,
        "peer_comparison": peer_comparison,
        "retrieval_pass": retrieval_pass,
        **assessment,
    }


def _save_report(report):
    os.makedirs(ESCALATION_DIR, exist_ok=True)
    safe_timestamp = report["generated_at"].replace(":", "-")
    filename = f"account_{report['account_id']}_{safe_timestamp}.json"
    path = os.path.join(ESCALATION_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def _log_episodic_memory(report, chroma_client=None):
    client = chroma_client or chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name="fraud_cases")

    summary = (
        f"Account {report['account_id']} ({report['risk_tier']} risk, "
        f"fraud probability {report['fraud_probability']:.4f}): {report['investigation']['root_cause']} "
        f"Final decision: {report['final_decision']}."
    )

    collection.upsert(
        documents=[summary],
        metadatas=[{
            "account_id": report["account_id"],
            "risk_tier": report["risk_tier"],
            "fraud_probability": report["fraud_probability"],
            "final_decision": report["final_decision"],
            "generated_at": report["generated_at"],
        }],
        ids=[f"case_{report['account_id']}_{report['generated_at']}"],
    )


def escalation_agent(state: FraudCaseState, chroma_client=None) -> Dict[str, Any]:
    """Escalation node: applies the deterministic escalation rule, writes an audit-ready
    JSON report, and logs episodic memory.
    """
    account_id = state["account_id"]
    profile = state["profile"]

    rule_based = make_escalation_decision(
        risk_tier=profile["risk_tier"],
        fraud_probability=profile["fraud_probability"],
        regulatory_flags=state.get("regulatory_flags", []),
    )

    report = {
        "account_id": account_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "risk_tier": profile["risk_tier"],
        "fraud_probability": profile["fraud_probability"],
        "investigation": {
            "root_cause": state.get("root_cause"),
            "confidence": state.get("confidence"),
            "regulatory_flags": state.get("regulatory_flags", []),
            "ai_recommended_action": state.get("recommended_action"),
            "analyst_question": state.get("analyst_question"),
            "initial_assessment": state.get("initial_assessment"),
            "critique": state.get("critique"),
            "refined_assessment": state.get("refined_assessment"),
            "retrieval_passes": state.get("retrieval_pass", 1),
        },
        "rule_based_decision": rule_based["decision"],
        "rule_based_reason": rule_based["reason"],
        "final_decision": rule_based["decision"],
    }

    report_path = _save_report(report)
    report["_report_path"] = report_path

    _log_episodic_memory(report, chroma_client)

    return {"report": report}


def route_after_triage(state: FraudCaseState) -> str:
    return "investigation" if state.get("routing_decision") == "INVESTIGATE" else END


def route_after_investigation(state: FraudCaseState) -> str:
    """Conditional edge: below-threshold confidence sends the case back to Investigation
    for another retrieval pass; otherwise proceed to Escalation. MAX_RETRIEVAL_PASSES bounds
    the loop so a persistently low confidence score can't retry forever.
    """
    if state["confidence_score"] < CONFIDENCE_THRESHOLD and state["retrieval_pass"] < MAX_RETRIEVAL_PASSES:
        return "investigation"
    return "escalation"


def _build_graph():
    graph = StateGraph(FraudCaseState)
    graph.add_node("triage", triage_agent)
    graph.add_node("investigation", investigation_agent)
    graph.add_node("escalation", escalation_agent)

    graph.add_edge(START, "triage")
    graph.add_conditional_edges("triage", route_after_triage, {"investigation": "investigation", END: END})
    graph.add_conditional_edges(
        "investigation", route_after_investigation, {"investigation": "investigation", "escalation": "escalation"}
    )
    graph.add_edge("escalation", END)

    return graph.compile()


fraud_investigation_graph = _build_graph()


if __name__ == "__main__":
    from tools import _load_scored_accounts

    scored = _load_scored_accounts()
    sample_id = int(scored.loc[scored["risk_tier"] == "HIGH", "account_index"].iloc[0])

    print("=== Fraud Investigation Graph ===")
    final_state = fraud_investigation_graph.invoke({"account_id": sample_id})
    print(json.dumps({k: v for k, v in final_state.items() if k != "profile"}, indent=2, default=str))
