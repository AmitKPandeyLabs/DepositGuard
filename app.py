import glob
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SCORED_ACCOUNTS_PATH = os.path.join(OUTPUTS_DIR, "scored_accounts.csv")
MODEL_COMPARISON_PATH = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
FEATURE_IMPORTANCE_IMG = os.path.join(OUTPUTS_DIR, "feature_importance.png")
FEATURED_DATA_SAMPLE_PATH = os.path.join(OUTPUTS_DIR, "featured_data_sample.csv")
ESCALATION_REPORTS_DIR = os.path.join(BASE_DIR, "rag_agent", "outputs", "escalation_reports")

RISK_COLORS = {"HIGH": "#ff6b6b", "MEDIUM": "#ffd93d", "LOW": "#4ecdc4"}
PLOTLY_TEMPLATE = "plotly_dark"

# CSS-enforced pixel height for the Fraud Dashboard's side-by-side SHAP chart / accounts table
# panels (see the matching stylesheet rule below) — both cards are forced to this exact height
# regardless of their internal content, so their borders always end flush.
SIDE_BY_SIDE_PANEL_HEIGHT = 320

st.set_page_config(page_title="DepositGuard", page_icon="\U0001F6E1", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .main-subtitle {
        color: #9aa0a6;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px 6px 16px;
    }
    /* Tighter rhythm for the dense dashboard grid of mini panels */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.75rem;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }
    div[data-testid="stPlotlyChart"] {
        margin: 0;
    }
    /* Card-style wrapper for mini panels, matching the stMetric border/background treatment */
    div[class*="st-key-mini-panel-"] {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px 6px 16px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Force the Top 5 SHAP Features / Top 5 highest-risk accounts panels (row 2 of the Fraud
# Dashboard's "At a Glance" grid) to an identical fixed pixel height via CSS, so both cards
# end flush regardless of chart vs. table content — independent of either panel's own rendering.
st.markdown(
    f"""
    <style>
    div[class*="st-key-mini-panel-shap"],
    div[class*="st-key-mini-panel-top-accounts"] {{
        height: {SIDE_BY_SIDE_PANEL_HEIGHT}px;
        overflow: hidden;
        box-sizing: border-box;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Cached data loaders
# --------------------------------------------------------------------------------------

@st.cache_data
def load_scored_accounts():
    return pd.read_csv(SCORED_ACCOUNTS_PATH)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(MODEL_COMPARISON_PATH, index_col=0)


@st.cache_data
def load_featured_columns(columns):
    scored = load_scored_accounts()
    account_ids = scored["account_index"].tolist()
    df = pd.read_csv(FEATURED_DATA_SAMPLE_PATH, usecols=["account_index", *columns]).set_index("account_index")
    return df.loc[account_ids]


@st.cache_data
def compute_shap_importance(top_n=10):
    df = load_scored_accounts()
    frames = []
    for i in range(1, 6):
        sub = df[[f"shap_feature_{i}", f"shap_value_{i}"]].rename(
            columns={f"shap_feature_{i}": "feature", f"shap_value_{i}": "value"}
        )
        frames.append(sub)
    long_df = pd.concat(frames, ignore_index=True)
    long_df["abs_value"] = long_df["value"].abs()
    importance = (
        long_df.groupby("feature")["abs_value"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return importance


def find_escalation_reports(account_id):
    pattern = os.path.join(ESCALATION_REPORTS_DIR, f"account_{account_id}_*.json")
    return sorted(glob.glob(pattern))


def missing_file_notice(path, label):
    st.warning(f"**{label}** not found at `{os.path.relpath(path, BASE_DIR)}`. Run the corresponding notebook/script to generate it.")


FEATURE_BIN_OPTIONS = {
    "Credit Risk Score": "credit_risk_score",
    "Customer Age": "customer_age",
    "Income": "income",
    "Velocity (6h)": "velocity_6h",
}


# --------------------------------------------------------------------------------------
# Mini panel renderers (shared between the dashboard grid and the full-size pages)
# --------------------------------------------------------------------------------------

def render_mini_model_comparison():
    if not os.path.exists(MODEL_COMPARISON_PATH):
        missing_file_notice(MODEL_COMPARISON_PATH, "outputs/model_comparison.csv")
        return

    comparison = load_model_comparison()
    winner = comparison["AUC-ROC"].idxmax()
    colors = ["gold" if model == winner else "rgba(0,0,0,0)" for model in comparison.index]
    widths = [2 if model == winner else 0 for model in comparison.index]

    fig = px.bar(
        x=comparison.index,
        y=comparison["AUC-ROC"],
        template=PLOTLY_TEMPLATE,
        labels={"x": "", "y": "AUC-ROC"},
        text_auto=".3f",
    )
    fig.update_traces(marker_line_color=colors, marker_line_width=widths)
    fig.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")


def render_mini_shap_importance(top_n=5, height=220):
    importance = compute_shap_importance(top_n=top_n)
    fig = px.bar(
        x=importance.values,
        y=importance.index,
        orientation="h",
        template=PLOTLY_TEMPLATE,
        labels={"x": "", "y": ""},
        color=importance.values,
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, width="stretch")


def get_sorted_high_risk_accounts():
    """HIGH-risk accounts sorted by fraud_probability descending — the single source of
    truth for "top risk" ordering, shared by the dashboard's top-accounts panel and the
    Case review page's default account selection.
    """
    df = load_scored_accounts()
    return df[df["risk_tier"] == "HIGH"].sort_values("fraud_probability", ascending=False)


def render_top_high_risk_table(n=3):
    top = get_sorted_high_risk_accounts().head(n)

    if top.empty:
        st.info("No HIGH risk accounts found.")
        return

    rows_html = ""
    for _, row in top.iterrows():
        color = RISK_COLORS.get(row["risk_tier"], "#9aa0a6")
        badge = (
            f'<span style="background-color:{color}22; color:{color}; border:1px solid {color}; '
            f'border-radius:6px; padding:2px 10px; font-size:0.8rem; font-weight:600;">{row["risk_tier"]}</span>'
        )
        rows_html += (
            "<tr style='border-bottom:1px solid rgba(255,255,255,0.06);'>"
            f"<td style='padding:10px 8px 10px 0;'>{int(row['account_index'])}</td>"
            f"<td style='padding:10px 8px;'>{row['fraud_probability']:.4f}</td>"
            f"<td style='padding:10px 0;'>{badge}</td></tr>"
        )

    st.markdown(
        f"""
        <table style="width:100%; font-size:0.9rem; border-collapse:collapse;">
        <thead>
            <tr style="color:#9aa0a6; text-align:left; border-bottom:1px solid rgba(255,255,255,0.12);">
                <th style="padding:8px 8px 8px 0;">Account</th>
                <th style="padding:8px 8px;">Fraud Prob.</th>
                <th style="padding:8px 0;">Risk</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_mini_risk_donut():
    df = load_scored_accounts()
    tier_counts = df["risk_tier"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0).astype(int)

    fig = px.pie(
        names=tier_counts.index,
        values=tier_counts.values,
        color=tier_counts.index,
        color_discrete_map=RISK_COLORS,
        hole=0.55,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(textinfo="percent")
    fig.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def render_mini_fraud_trend(bins=40, smoothing_window=3):
    """Fraud probability distribution as a smoothed area chart (same data as the histogram)."""
    df = load_scored_accounts()
    counts, bin_edges = np.histogram(df["fraud_probability"], bins=bins)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    smoothed = pd.Series(counts).rolling(window=smoothing_window, min_periods=1, center=True).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bin_mids,
        y=smoothed,
        mode="lines",
        fill="tozeroy",
        line=dict(color="#4ecdc4", shape="spline"),
    ))
    fig.add_vline(x=0.40, line_dash="dash", line_color="#ffd93d")
    fig.add_vline(x=0.75, line_dash="dash", line_color="#ff6b6b")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title=None,
        yaxis_title=None,
        height=220,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------------------
# Page 1 - Fraud Dashboard
# --------------------------------------------------------------------------------------

def page_fraud_dashboard():
    st.header("Fraud Dashboard")

    if not os.path.exists(SCORED_ACCOUNTS_PATH):
        missing_file_notice(SCORED_ACCOUNTS_PATH, "outputs/scored_accounts.csv")
        return

    df = load_scored_accounts()
    total = len(df)
    tier_counts = df["risk_tier"].value_counts().reindex(["HIGH", "MEDIUM", "LOW"]).fillna(0).astype(int)
    fraud_rate = (df["risk_tier"] == "HIGH").mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Accounts Scored", f"{total:,}")
    c2.metric("Predicted Fraud Rate (HIGH tier)", f"{fraud_rate:.2f}%")
    c3.metric("HIGH Risk", f"{tier_counts['HIGH']:,}")
    c4.metric("MEDIUM Risk", f"{tier_counts['MEDIUM']:,}")
    c5.metric("LOW Risk", f"{tier_counts['LOW']:,}")

    st.caption(
        "This sample (`scored_accounts.csv`) has no ground-truth fraud label, so \"fraud rate\" here "
        "is the model's predicted HIGH-risk rate, not a confirmed fraud incidence rate."
    )

    st.subheader("At a Glance")

    donut_col, model_col, trend_col = st.columns([1, 1, 2])
    with donut_col:
        with st.container(key="mini-panel-donut"):
            st.markdown("**Risk Tier Distribution**")
            render_mini_risk_donut()
    with model_col:
        with st.container(key="mini-panel-model"):
            st.markdown("**Model Comparison (AUC-ROC)**")
            render_mini_model_comparison()
    with trend_col:
        with st.container(key="mini-panel-trend"):
            st.markdown("**Fraud Probability Distribution**")
            render_mini_fraud_trend()

    shap_col, accounts_col = st.columns(2)
    with shap_col:
        with st.container(key="mini-panel-shap"):
            st.markdown("**Top 5 SHAP Features**")
            render_mini_shap_importance(top_n=5, height=260)
    with accounts_col:
        with st.container(key="mini-panel-top-accounts"):
            st.markdown("**Top 5 highest-risk accounts**")
            render_top_high_risk_table(n=5)


# --------------------------------------------------------------------------------------
# Page 2 - Model Insights (Model Performance + Feature Importance)
# --------------------------------------------------------------------------------------

def page_model_insights():
    st.header("Model Insights")

    comparison = None
    if not os.path.exists(MODEL_COMPARISON_PATH):
        missing_file_notice(MODEL_COMPARISON_PATH, "outputs/model_comparison.csv")
    else:
        comparison = load_model_comparison()
        metrics = ["AUC-ROC", "Precision", "Recall", "F1"]
        winner = comparison["AUC-ROC"].idxmax()

        long_df = comparison[metrics].reset_index().melt(id_vars=comparison.index.name or "index", var_name="Metric", value_name="Score")
        long_df = long_df.rename(columns={comparison.index.name or "index": "Model"})

        fig = px.bar(
            long_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            template=PLOTLY_TEMPLATE,
            text_auto=".3f",
        )
        fig.update_layout(yaxis_title="Score", xaxis_title="", legend_title="Metric", margin=dict(t=20, b=20))

        for trace in fig.data:
            # Outline the winning model's bars so it stands out across every metric group.
            colors = ["gold" if model == winner else "rgba(0,0,0,0)" for model in long_df[long_df["Metric"] == trace.name]["Model"]]
            widths = [3 if model == winner else 0 for model in long_df[long_df["Metric"] == trace.name]["Model"]]
            trace.marker.line.color = colors
            trace.marker.line.width = widths

        st.plotly_chart(fig, width="stretch")

        winner_row = comparison.loc[winner]
        st.success(
            f"\U0001F3C6 **{winner}** wins on AUC-ROC ({winner_row['AUC-ROC']:.4f}) - the standard threshold-independent "
            f"metric for imbalanced fraud classification. Precision {winner_row['Precision']:.4f}, "
            f"Recall {winner_row['Recall']:.4f}, F1 {winner_row['F1']:.4f}, FPR {winner_row['FPR']:.4f}."
        )

    st.divider()

    # Row 2: SHAP image (left, unchanged position) + recomputed Top 10 SHAP chart (right)
    image_col, shap_col = st.columns(2)

    with image_col:
        st.subheader("Top fraud signals")
        if os.path.exists(FEATURE_IMPORTANCE_IMG):
            st.image(FEATURE_IMPORTANCE_IMG, width=550)
        else:
            missing_file_notice(FEATURE_IMPORTANCE_IMG, "outputs/feature_importance.png")

    with shap_col:
        st.subheader("Signal breakdown")
        st.caption(
            "Mean |SHAP value|, aggregated only over rows where a feature appears in that account's top-5 "
            "signals (scored_accounts.csv only stores each account's top 5) - a slightly different, but "
            "consistent, cross-check against the plot above."
        )

        if not os.path.exists(SCORED_ACCOUNTS_PATH):
            missing_file_notice(SCORED_ACCOUNTS_PATH, "outputs/scored_accounts.csv")
        else:
            importance = compute_shap_importance(top_n=10)
            fig2 = px.bar(
                x=importance.values,
                y=importance.index,
                orientation="h",
                template=PLOTLY_TEMPLATE,
                labels={"x": "Mean |SHAP value|", "y": "Feature"},
                color=importance.values,
                color_continuous_scale="Reds",
            )
            fig2.update_layout(
                yaxis=dict(autorange="reversed"),
                coloraxis_showscale=False,
                height=600,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig2, width="stretch")

    st.divider()

    # Row 3: full comparison table, full width
    st.subheader("Model comparison")
    if comparison is not None:
        st.dataframe(
            comparison.style.highlight_max(axis=0, color="rgba(255, 215, 0, 0.25)"),
            width="stretch",
        )


# --------------------------------------------------------------------------------------
# Page 3 - Case Review (Fraud Investigation + Fraud Trend Analysis)
# --------------------------------------------------------------------------------------

def resolve_default_case_review_account():
    """The #1 highest-fraud_probability HIGH-risk account (same ordering as the dashboard's
    top-accounts panel) — falling back to the next-highest account that actually has a
    generated escalation report if the true #1 doesn't have one.
    """
    ranked_ids = get_sorted_high_risk_accounts()["account_index"].tolist()
    for account_id in ranked_ids:
        if find_escalation_reports(account_id):
            return account_id
    return ranked_ids[0] if ranked_ids else None


def page_case_review():
    st.header("Case Review")

    if not os.path.exists(SCORED_ACCOUNTS_PATH):
        missing_file_notice(SCORED_ACCOUNTS_PATH, "outputs/scored_accounts.csv")
        return

    df = load_scored_accounts()
    high_risk_ids = sorted(df.loc[df["risk_tier"] == "HIGH", "account_index"].tolist())

    if not high_risk_ids:
        st.info("No HIGH risk accounts found in scored_accounts.csv.")
    else:
        default_account = resolve_default_case_review_account()
        default_index = (
            high_risk_ids.index(default_account)
            if default_account in high_risk_ids
            else 0
        )
        account_id = st.selectbox("Select a HIGH risk account", high_risk_ids, index=default_index)
        row = df.loc[df["account_index"] == account_id].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Account ID", account_id)
        c2.metric("Fraud Probability", f"{row['fraud_probability']:.4f}")
        c3.metric("Risk Tier", row["risk_tier"])

        st.subheader("Top SHAP Features")
        shap_rows = [
            {"Feature": row[f"shap_feature_{i}"], "SHAP Value": row[f"shap_value_{i}"]}
            for i in range(1, 6)
        ]
        shap_df = pd.DataFrame(shap_rows)

        fig = px.bar(
            shap_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            template=PLOTLY_TEMPLATE,
            color="SHAP Value",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Escalation Report")
        report_paths = find_escalation_reports(account_id)

        if not report_paths:
            st.info(
                "No escalation report found for this account in `rag_agent/outputs/escalation_reports/`. "
                "Run `rag_agent/main.py` to investigate more HIGH risk accounts."
            )
        else:
            if len(report_paths) > 1:
                st.caption(f"{len(report_paths)} reports found for this account - showing the most recent.")

            with open(report_paths[-1], encoding="utf-8") as f:
                report = json.load(f)

            inv = report.get("investigation", {})

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"**Confidence:** {inv.get('confidence', 'n/a')}")
                st.markdown(f"**Final Decision:** {report.get('final_decision', 'n/a')}")
            with rc2:
                st.markdown(f"**Rule-Based Reason:** {report.get('rule_based_reason', 'n/a')}")

            st.markdown(f"**Root Cause:** {inv.get('root_cause', 'n/a')}")

            flags = inv.get("regulatory_flags", [])
            if flags:
                st.markdown("**Regulatory Flags:**")
                for flag in flags:
                    st.markdown(f"- {flag}")

            st.markdown(f"**Analyst Question:** {inv.get('analyst_question', 'n/a')}")
            st.markdown(f"**Critique:** {inv.get('critique', 'n/a')}")

            with st.expander("Full escalation report (JSON)"):
                st.json(report)

    st.divider()

    st.subheader("Fraud Probability by Feature Bin")
    if os.path.exists(FEATURED_DATA_SAMPLE_PATH):
        feature_label = st.selectbox("Bin by feature", list(FEATURE_BIN_OPTIONS.keys()))
        feature_col = FEATURE_BIN_OPTIONS[feature_label]

        featured = load_featured_columns((feature_col,))
        merged = df[["account_index", "fraud_probability"]].join(featured, on="account_index")
        merged["bin"] = pd.qcut(merged[feature_col], q=10, duplicates="drop")
        merged["bin_mid"] = merged["bin"].apply(lambda b: b.mid).astype(float)

        binned = merged.groupby("bin_mid", observed=True)["fraud_probability"].mean().reset_index()
        binned = binned.sort_values("bin_mid")

        fig3 = px.line(
            binned,
            x="bin_mid",
            y="fraud_probability",
            markers=True,
            template=PLOTLY_TEMPLATE,
            labels={"bin_mid": feature_label, "fraud_probability": "Mean Fraud Probability"},
        )
        fig3.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig3, width="stretch")
    else:
        missing_file_notice(FEATURED_DATA_SAMPLE_PATH, "outputs/featured_data_sample.csv")

    st.subheader("Distribution of Fraud Probabilities")
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=df["fraud_probability"], nbinsx=60, marker_color="#4ecdc4"))
    fig4.add_vline(x=0.40, line_dash="dash", line_color="#ffd93d", annotation_text="MEDIUM (0.40)")
    fig4.add_vline(x=0.75, line_dash="dash", line_color="#ff6b6b", annotation_text="HIGH (0.75)")
    fig4.update_layout(
        template=PLOTLY_TEMPLATE,
        xaxis_title="Fraud Probability",
        yaxis_title="Account Count",
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig4, width="stretch")


# --------------------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------------------

PAGES = {
    "Fraud dashboard": page_fraud_dashboard,
    "Model insights": page_model_insights,
    "Case review": page_case_review,
}

st.sidebar.title("\U0001F6E1️ DepositGuard")
st.sidebar.caption("Deposit Account Fraud Detection")
selection = st.sidebar.radio("Navigate", list(PAGES.keys()))

st.markdown('<p class="main-title">\U0001F6E1️ DepositGuard</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Deposit account fraud detection - model performance, explainability, and case investigation</p>', unsafe_allow_html=True)

PAGES[selection]()
