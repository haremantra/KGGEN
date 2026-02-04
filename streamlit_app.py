"""KGGEN-CUAD Portfolio Analysis Dashboard.

A Streamlit web application for contract analysis and portfolio risk management.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import json

# Configuration
API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="KGGEN Contract Analyzer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .risk-critical { background-color: #ff4444; color: white; padding: 5px 10px; border-radius: 4px; }
    .risk-high { background-color: #ff8800; color: white; padding: 5px 10px; border-radius: 4px; }
    .risk-medium { background-color: #ffcc00; color: black; padding: 5px 10px; border-radius: 4px; }
    .risk-low { background-color: #44aa44; color: white; padding: 5px 10px; border-radius: 4px; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def api_request(method: str, endpoint: str, **kwargs) -> Optional[dict]:
    """Make an API request and handle errors."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def risk_badge(level: str) -> str:
    """Create HTML badge for risk level."""
    css_class = f"risk-{level.lower()}"
    return f'<span class="{css_class}">{level}</span>'


# Sidebar Navigation
st.sidebar.title("📜 KGGEN Analyzer")
page = st.sidebar.radio(
    "Navigation",
    ["Upload", "Portfolio", "Analysis", "Compare", "Gaps"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Stats")

# Try to load contracts for sidebar stats
contracts = api_request("GET", "/contracts")
if contracts:
    st.sidebar.metric("Contracts", len(contracts))
    analyzed = sum(1 for c in contracts if c.get("risk_score") is not None)
    st.sidebar.metric("Analyzed", analyzed)
    if analyzed > 0:
        avg_risk = sum(c.get("risk_score", 0) for c in contracts if c.get("risk_score")) / analyzed
        st.sidebar.metric("Avg Risk Score", f"{avg_risk:.0f}/100")


# === Page: Upload ===
if page == "Upload":
    st.title("📤 Upload Contracts")
    st.markdown("Upload contract PDFs or text files for analysis.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Choose contract files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload PDF or text files containing contract content",
        )

        if uploaded_files:
            st.markdown(f"**Selected {len(uploaded_files)} file(s)**")

            if st.button("Analyze All", type="primary"):
                progress = st.progress(0)
                status = st.empty()

                for i, file in enumerate(uploaded_files):
                    status.text(f"Processing: {file.name}")

                    files = {"file": (file.name, file.getvalue(), "application/pdf")}
                    result = api_request("POST", "/contracts/upload", files=files)

                    if result:
                        st.success(f"✅ {file.name} - Risk Score: {result.get('risk_score', 'N/A')}")
                    else:
                        st.error(f"❌ {file.name} - Failed")

                    progress.progress((i + 1) / len(uploaded_files))

                status.text("Complete!")
                st.balloons()

    with col2:
        st.markdown("### Uploaded Contracts")
        if contracts:
            for c in contracts[-5:]:  # Show last 5
                risk_level = c.get("risk_level", "N/A")
                st.markdown(
                    f"**{c['filename'][:20]}...** "
                    f"({c['contract_id']}) - {risk_badge(risk_level)}",
                    unsafe_allow_html=True
                )
        else:
            st.info("No contracts uploaded yet")


# === Page: Portfolio ===
elif page == "Portfolio":
    st.title("📊 Portfolio Overview")

    if not contracts or len(contracts) == 0:
        st.warning("No contracts uploaded. Go to Upload page to add contracts.")
    else:
        # Get portfolio risks
        portfolio_risks = api_request("GET", "/portfolio/risks")

        if portfolio_risks:
            # Top metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Contracts", portfolio_risks["total_contracts"])
            with col2:
                st.metric("Avg Risk Score", f"{portfolio_risks['average_risk_score']:.0f}/100")
            with col3:
                critical = portfolio_risks["contracts_by_risk_level"].get("CRITICAL", 0)
                st.metric("Critical Risk", critical, delta=None)
            with col4:
                high = portfolio_risks["contracts_by_risk_level"].get("HIGH", 0)
                st.metric("High Risk", high, delta=None)

            st.markdown("---")

            # Risk Distribution Chart
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Risk Distribution")
                risk_levels = portfolio_risks["contracts_by_risk_level"]
                if risk_levels:
                    fig = px.pie(
                        values=list(risk_levels.values()),
                        names=list(risk_levels.keys()),
                        color=list(risk_levels.keys()),
                        color_discrete_map={
                            "CRITICAL": "#ff4444",
                            "HIGH": "#ff8800",
                            "MEDIUM": "#ffcc00",
                            "LOW": "#44aa44",
                        },
                        hole=0.4,
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Most Common Risks")
                common_risks = portfolio_risks.get("most_common_risks", [])
                if common_risks:
                    df = pd.DataFrame(common_risks)
                    fig = px.bar(
                        df,
                        x="count",
                        y="label",
                        orientation="h",
                        color="count",
                        color_continuous_scale="Reds",
                    )
                    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Occurrences")
                    st.plotly_chart(fig, use_container_width=True)

            # Contract Risk Heatmap
            st.subheader("Contract Risk Heatmap")
            if contracts:
                df = pd.DataFrame(contracts)
                if "risk_score" in df.columns:
                    df = df.dropna(subset=["risk_score"])
                    if len(df) > 0:
                        fig = px.treemap(
                            df,
                            path=["risk_level", "contract_id"],
                            values="risk_score",
                            color="risk_score",
                            color_continuous_scale="RdYlGn_r",
                            title="",
                        )
                        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)

            # Highest Risk Contracts Table
            st.subheader("Highest Risk Contracts")
            highest_risk = portfolio_risks.get("highest_risk_contracts", [])
            if highest_risk and contracts:
                high_risk_data = [
                    c for c in contracts
                    if c["contract_id"] in highest_risk
                ]
                if high_risk_data:
                    df = pd.DataFrame(high_risk_data)
                    st.dataframe(
                        df[["contract_id", "filename", "risk_score", "risk_level"]],
                        use_container_width=True,
                        hide_index=True,
                    )


# === Page: Analysis ===
elif page == "Analysis":
    st.title("🔍 Contract Analysis")

    if not contracts:
        st.warning("No contracts uploaded. Go to Upload page to add contracts.")
    else:
        # Contract selector
        contract_options = {f"{c['filename']} ({c['contract_id']})": c['contract_id'] for c in contracts}
        selected = st.selectbox("Select Contract", list(contract_options.keys()))

        if selected:
            contract_id = contract_options[selected]

            # Load analysis and risks
            analysis = api_request("GET", f"/contracts/{contract_id}")
            risks = api_request("GET", f"/contracts/{contract_id}/risks")

            if analysis and risks:
                # Overview metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Risk Score", f"{risks['overall_risk_score']}/100")
                with col2:
                    st.markdown(f"**Risk Level:** {risk_badge(risks['risk_level'])}", unsafe_allow_html=True)
                with col3:
                    st.metric("Clauses Analyzed", analysis["total_clauses"])
                with col4:
                    st.metric("Labels Found", len(analysis["analyzed_clauses"]))

                st.markdown("---")

                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["Risk Findings", "Clause Details", "LLM Analysis"])

                with tab1:
                    st.subheader("Risk Findings")

                    # Present clause risks
                    findings = risks.get("findings", [])
                    if findings:
                        for f in findings:
                            severity = f["severity"]
                            with st.expander(f"{risk_badge(severity)} {f['label']}", expanded=severity in ["CRITICAL", "HIGH"]):
                                st.markdown(f"**Reason:** {f['reason']}")
                                st.markdown(f"**Recommendation:** {f['recommendation']}")
                                if f.get("clause_text"):
                                    st.text_area("Clause Text", f["clause_text"], height=100, disabled=True)
                    else:
                        st.info("No risk findings for present clauses")

                    # Missing clause risks
                    st.subheader("Missing Protections")
                    missing = risks.get("missing_clause_risks", [])
                    if missing:
                        for f in missing:
                            severity = f["severity"]
                            with st.expander(f"{risk_badge(severity)} Missing: {f['label']}"):
                                st.markdown(f"**Why it matters:** {f['reason']}")
                                st.markdown(f"**Recommendation:** {f['recommendation']}")
                    else:
                        st.success("All standard protections present!")

                with tab2:
                    st.subheader("Analyzed Clauses")

                    # Group by category
                    clauses = analysis.get("analyzed_clauses", [])
                    if clauses:
                        categories = {}
                        for c in clauses:
                            cat = c.get("category", "Other")
                            if cat not in categories:
                                categories[cat] = []
                            categories[cat].append(c)

                        for category, items in categories.items():
                            st.markdown(f"### {category.replace('_', ' ').title()}")
                            for item in items:
                                conf_pct = f"{item['confidence']:.0%}"
                                with st.expander(f"{item['cuad_label']} ({conf_pct})"):
                                    st.text_area("Text", item.get("text_preview", ""), height=80, disabled=True)

                                    if item.get("extracted_values"):
                                        st.markdown("**Extracted Values:**")
                                        for v in item["extracted_values"]:
                                            st.markdown(f"- **{v['field']}:** {v['value']} ({v['confidence']:.0%})")

                                    if item.get("entities"):
                                        st.markdown(f"**Entities:** {', '.join(item['entities'])}")
                    else:
                        st.info("No clauses analyzed")

                with tab3:
                    st.subheader("AI Analysis")
                    if risks.get("llm_analysis"):
                        st.markdown(risks["llm_analysis"])
                    else:
                        st.info("LLM analysis not available for low-risk contracts")

                    st.subheader("Summary")
                    st.markdown(risks.get("summary", "No summary available"))


# === Page: Compare ===
elif page == "Compare":
    st.title("⚖️ Compare Contracts")

    if not contracts or len(contracts) < 2:
        st.warning("Need at least 2 contracts to compare. Upload more contracts.")
    else:
        col1, col2 = st.columns(2)

        contract_options = {f"{c['filename']} ({c['contract_id']})": c['contract_id'] for c in contracts}

        with col1:
            selected_a = st.selectbox("Contract A", list(contract_options.keys()), key="compare_a")
        with col2:
            selected_b = st.selectbox("Contract B", list(contract_options.keys()), key="compare_b", index=min(1, len(contract_options)-1))

        if selected_a and selected_b and selected_a != selected_b:
            contract_a = contract_options[selected_a]
            contract_b = contract_options[selected_b]

            if st.button("Compare", type="primary"):
                comparison = api_request(
                    "POST",
                    "/portfolio/compare",
                    json={"contract_a": contract_a, "contract_b": contract_b}
                )

                if comparison:
                    # Risk comparison
                    st.subheader("Risk Comparison")
                    col1, col2, col3 = st.columns(3)

                    risk_comp = comparison["risk_comparison"]
                    with col1:
                        st.metric(
                            "Contract A",
                            f"{risk_comp[contract_a]['score']}/100",
                            delta=None,
                        )
                        st.markdown(f"Level: {risk_badge(risk_comp[contract_a]['level'])}", unsafe_allow_html=True)
                    with col2:
                        diff = risk_comp["difference"]
                        st.metric("Difference", f"{diff:+d}")
                    with col3:
                        st.metric(
                            "Contract B",
                            f"{risk_comp[contract_b]['score']}/100",
                            delta=None,
                        )
                        st.markdown(f"Level: {risk_badge(risk_comp[contract_b]['level'])}", unsafe_allow_html=True)

                    st.markdown("---")

                    # Clause comparison
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("Only in A")
                        for label in comparison["only_in_a"]:
                            st.markdown(f"- {label}")
                        if not comparison["only_in_a"]:
                            st.info("None")

                    with col2:
                        st.subheader("Shared")
                        for label in comparison["shared_clauses"]:
                            st.markdown(f"- {label}")
                        if not comparison["shared_clauses"]:
                            st.info("None")

                    with col3:
                        st.subheader("Only in B")
                        for label in comparison["only_in_b"]:
                            st.markdown(f"- {label}")
                        if not comparison["only_in_b"]:
                            st.info("None")

                    # Clause value differences
                    if comparison.get("clause_differences"):
                        st.subheader("Clause Value Differences")
                        for diff in comparison["clause_differences"]:
                            with st.expander(diff["label"]):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Contract A:**")
                                    st.json(diff.get(contract_a, {}))
                                with col2:
                                    st.markdown("**Contract B:**")
                                    st.json(diff.get(contract_b, {}))


# === Page: Gaps ===
elif page == "Gaps":
    st.title("🔎 Gap Analysis")
    st.markdown("Identify missing standard protections across your contract portfolio.")

    if not contracts:
        st.warning("No contracts uploaded. Go to Upload page to add contracts.")
    else:
        gaps = api_request("GET", "/portfolio/gaps")

        if gaps:
            # Summary metrics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Standard Protections Checked")
                for protection in gaps.get("standard_protections", []):
                    st.markdown(f"- {protection}")

            with col2:
                st.subheader("Gap Frequency")
                gap_freq = gaps.get("gap_frequency", [])
                if gap_freq:
                    df = pd.DataFrame(gap_freq)
                    fig = px.bar(
                        df,
                        x="missing_percentage",
                        y="label",
                        orientation="h",
                        color="missing_percentage",
                        color_continuous_scale="Reds",
                        labels={"missing_percentage": "% Missing", "label": "Clause"},
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Priority recommendations
            st.subheader("Priority Recommendations")
            recommendations = gaps.get("priority_recommendations", [])
            if recommendations:
                for rec in recommendations:
                    priority = rec["priority"]
                    color = "red" if priority == "HIGH" else "orange"
                    with st.expander(f"🔴 {rec['label']} - Missing in {rec['missing_percentage']}% of contracts"):
                        st.markdown(f"**Recommendation:** {rec['recommendation']}")
                        st.markdown(f"**Affected Contracts:** {len(rec['affected_contracts'])}")
                        with st.container():
                            for cid in rec["affected_contracts"][:10]:
                                st.markdown(f"- {cid}")
                            if len(rec["affected_contracts"]) > 10:
                                st.markdown(f"... and {len(rec['affected_contracts']) - 10} more")
            else:
                st.success("No critical gaps identified!")

            # Detailed gap by contract
            st.subheader("Gaps by Contract")
            gap_by_contract = gaps.get("gap_by_contract", {})
            if gap_by_contract:
                # Create a matrix view
                all_labels = gaps.get("standard_protections", [])
                matrix_data = []

                for contract_id, missing in gap_by_contract.items():
                    row = {"Contract": contract_id}
                    for label in all_labels:
                        row[label] = "❌" if label in missing else "✅"
                    matrix_data.append(row)

                df = pd.DataFrame(matrix_data)
                st.dataframe(df, use_container_width=True, hide_index=True)


# Footer
st.markdown("---")
st.markdown(
    "<center>KGGEN-CUAD Contract Analyzer | "
    "Built with Streamlit & FastAPI</center>",
    unsafe_allow_html=True
)
