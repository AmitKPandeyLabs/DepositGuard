import os

import numpy as np
import pandas as pd

from knowledge_base import query_knowledge_base

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
SCORED_ACCOUNTS_PATH = os.path.join(OUTPUTS_DIR, "scored_accounts.csv")
FEATURED_DATA_PATH = os.path.join(OUTPUTS_DIR, "featured_data.csv")

# Raw account-detail columns surfaced to the investigation agent. `fraud_bool` (the
# ground-truth label) is deliberately excluded — the agent must reason from behavioral
# signals, not the answer key.
ACCOUNT_DETAIL_COLUMNS = [
    "income",
    "customer_age",
    "credit_risk_score",
    "payment_type",
    "employment_status",
    "housing_status",
    "device_os",
    "email_is_free",
    "proposed_credit_limit",
    "velocity_6h",
    "velocity_24h",
    "zip_count_4w",
    "days_since_request",
    "current_address_months_count",
    "month",
]

_scored_accounts_df = None
_featured_data_df = None


def _sanitize(value):
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_scored_accounts():
    global _scored_accounts_df
    if _scored_accounts_df is None:
        _scored_accounts_df = pd.read_csv(SCORED_ACCOUNTS_PATH)
    return _scored_accounts_df


def _load_featured_data():
    global _featured_data_df
    if _featured_data_df is None:
        if os.path.exists(FEATURED_DATA_PATH):
            _featured_data_df = pd.read_csv(FEATURED_DATA_PATH, usecols=ACCOUNT_DETAIL_COLUMNS)
        else:
            _featured_data_df = False
    return _featured_data_df if _featured_data_df is not False else None


def get_account_profile(account_id):
    """Read the account's fraud score, risk tier, and top SHAP signals from scored_accounts.csv."""
    account_id = int(account_id)
    scored = _load_scored_accounts()
    match = scored[scored["account_index"] == account_id]

    if match.empty:
        return {"error": f"account_id {account_id} not found in scored_accounts.csv"}

    row = match.iloc[0]

    top_shap_features = [
        {"feature": row[f"shap_feature_{i}"], "shap_value": float(row[f"shap_value_{i}"])}
        for i in range(1, 6)
    ]

    profile = {
        "account_id": account_id,
        "fraud_probability": float(row["fraud_probability"]),
        "risk_tier": row["risk_tier"],
        "top_shap_features": top_shap_features,
    }

    featured_df = _load_featured_data()
    if featured_df is not None and account_id in featured_df.index:
        profile["account_details"] = _sanitize(featured_df.loc[account_id].to_dict())

    return profile


def get_regulatory_context(fraud_type, n_results=4):
    """Query the ChromaDB fraud-regulation knowledge base for context relevant to a fraud type."""
    query = f"{fraud_type} regulations, investigation requirements, and reporting obligations"
    matches = query_knowledge_base(query, n_results=n_results)
    return {"fraud_type": fraud_type, "query": query, "matches": matches}


def get_peer_comparison(fraud_probability):
    """Compare a fraud probability against the scored-account population and return a percentile ranking."""
    fraud_probability = float(fraud_probability)
    scored = _load_scored_accounts()
    probs = scored["fraud_probability"].to_numpy()

    percentile = float((probs <= fraud_probability).mean() * 100)

    return {
        "fraud_probability": fraud_probability,
        "percentile": round(percentile, 2),
        "population_size": int(len(probs)),
        "population_mean": float(probs.mean()),
        "population_median": float(np.median(probs)),
        "population_p95": float(np.percentile(probs, 95)),
        "interpretation": (
            f"This account's fraud probability ({fraud_probability:.4f}) is higher than "
            f"{percentile:.1f}% of the {len(probs):,}-account scored sample."
        ),
    }


def make_escalation_decision(risk_tier, fraud_probability, regulatory_flags=None):
    """Rule-based escalation decision: CLEAR / MONITOR / ESCALATE / FREEZE."""
    regulatory_flags = regulatory_flags or []
    fraud_probability = float(fraud_probability)

    if risk_tier == "HIGH" and fraud_probability >= 0.90:
        decision = "FREEZE"
        reason = (
            f"HIGH risk tier with fraud probability {fraud_probability:.3f} >= 0.90 threshold — "
            "immediate account freeze pending manual review."
        )
    elif risk_tier == "HIGH":
        decision = "ESCALATE"
        reason = f"HIGH risk tier (fraud probability {fraud_probability:.3f}) — escalate to fraud investigation team."
    elif risk_tier == "MEDIUM" and regulatory_flags:
        decision = "ESCALATE"
        reason = (
            f"MEDIUM risk tier with {len(regulatory_flags)} regulatory flag(s) raised "
            f"({', '.join(regulatory_flags)}) — escalate for review."
        )
    elif risk_tier == "MEDIUM":
        decision = "MONITOR"
        reason = f"MEDIUM risk tier (fraud probability {fraud_probability:.3f}) with no regulatory flags — place on monitoring list."
    else:
        decision = "CLEAR"
        reason = f"LOW risk tier (fraud probability {fraud_probability:.3f}) — no action required."

    return {"decision": decision, "reason": reason}


if __name__ == "__main__":
    import json

    scored = _load_scored_accounts()
    sample_id = int(scored.loc[scored["risk_tier"] == "HIGH", "account_index"].iloc[0])

    print("=== get_account_profile ===")
    profile = get_account_profile(sample_id)
    print(json.dumps(profile, indent=2))

    print("\n=== get_regulatory_context ===")
    context = get_regulatory_context("new account fraud", n_results=2)
    print(json.dumps({**context, "matches": [{"source": m["source"], "distance": m["distance"]} for m in context["matches"]]}, indent=2))

    print("\n=== get_peer_comparison ===")
    print(json.dumps(get_peer_comparison(profile["fraud_probability"]), indent=2))

    print("\n=== make_escalation_decision ===")
    print(json.dumps(make_escalation_decision(profile["risk_tier"], profile["fraud_probability"], ["R10"]), indent=2))
