import json
import sys
import time

from agents import fraud_investigation_graph
from knowledge_base import build_knowledge_base
from tools import _load_scored_accounts

N_ACCOUNTS = 10


def run_pipeline():
    print("Building fraud regulation knowledge base...")
    collection = build_knowledge_base()
    print(f"  {collection.count()} chunks indexed in 'fraud_regulations'.\n")

    scored = _load_scored_accounts()
    high_risk = scored[scored["risk_tier"] == "HIGH"]
    top = high_risk.sort_values("fraud_probability", ascending=False).head(N_ACCOUNTS)
    account_ids = top["account_index"].astype(int).tolist()

    print(f"Loaded {len(account_ids)} HIGH risk accounts: {account_ids}\n")
    print("=" * 70)

    reports = []
    for i, account_id in enumerate(account_ids, start=1):
        print(f"[{i}/{len(account_ids)}] Account {account_id}")
        start = time.time()

        final_state = fraud_investigation_graph.invoke({"account_id": account_id})
        print(f"  Triage: {final_state['routing_decision']} — {final_state['triage_reason']}")

        if final_state["routing_decision"] != "INVESTIGATE":
            print("  Skipping investigation (not routed to Investigation Agent).")
            continue

        print(f"  Investigation: recommended_action={final_state['recommended_action']}, "
              f"confidence={final_state['confidence']} (retrieval passes: {final_state['retrieval_pass']})")

        report = final_state["report"]
        print(f"  Escalation: final_decision={report['final_decision']}")
        print(f"  Report saved: {report['_report_path']}")
        print(f"  Elapsed: {time.time() - start:.1f}s")
        print("-" * 70)

        reports.append(report)

    return reports


def print_summary(reports):
    print("\n" + "=" * 70)
    print(f"SUMMARY — {len(reports)} cases processed")
    print("=" * 70)

    header = f"{'Account':>10} {'Prob':>7} {'Confidence':>16} {'Rule Decision':>14}  Root Cause"
    print(header)
    print("-" * len(header))
    for r in reports:
        root_cause = (r["investigation"]["root_cause"] or "")[:55]
        print(
            f"{r['account_id']:>10} {r['fraud_probability']:>7.3f} "
            f"{r['investigation']['confidence']:>16} {r['final_decision']:>14}  {root_cause}"
        )

    decisions = [r["final_decision"] for r in reports]
    print("\nFinal decision breakdown (rule-based):")
    for decision in ["FREEZE", "ESCALATE", "MONITOR", "CLEAR"]:
        count = decisions.count(decision)
        if count:
            print(f"  {decision}: {count}")


def print_detailed_example(report):
    print("\n" + "=" * 70)
    print(f"DETAILED EXAMPLE — Account {report['account_id']}")
    print("=" * 70)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    reports = run_pipeline()

    if not reports:
        print("No cases were investigated.")
        sys.exit(0)

    print_summary(reports)
    print_detailed_example(reports[0])
