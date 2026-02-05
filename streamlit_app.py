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
    ["Upload", "Portfolio", "Analysis", "Compare", "Gaps", "Dependencies", "Search & QA"],
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


# === Page: Dependencies ===
elif page == "Dependencies":
    st.title("Dependencies")
    st.markdown("Analyze clause interdependencies within a contract.")

    # Select contract
    contracts = api_request("GET", "/contracts")
    if contracts:
        analyzed = [c for c in contracts if c.get("status") == "analyzed"]
        if not analyzed:
            st.info("No analyzed contracts. Upload and analyze a contract first.")
        else:
            options = {c["contract_id"]: f"{c['filename']} ({c['contract_id']})" for c in analyzed}
            selected_id = st.selectbox(
                "Select Contract",
                list(options.keys()),
                format_func=lambda x: options[x],
            )

            if selected_id:
                deps = api_request("GET", f"/contracts/{selected_id}/dependencies")

                if deps:
                    # Top metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Dependencies", len(deps.get("edges", [])))
                    col2.metric("Contradictions", deps.get("contradiction_count", 0))
                    col3.metric("Missing Requirements", len(deps.get("missing_requirements", [])))
                    col4.metric("Most Connected", deps.get("max_impact_clause", "N/A"))

                    st.markdown("---")

                    # Interactive network graph
                    st.subheader("Dependency Network")

                    nodes = deps.get("nodes", [])
                    edges = deps.get("edges", [])

                    if nodes and edges:
                        import networkx as nx

                        # Build graph for layout
                        G = nx.DiGraph()
                        for n in nodes:
                            G.add_node(n["label"])
                        for e in edges:
                            G.add_edge(e["source_label"], e["target_label"])

                        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

                        # Edge colors by dependency type
                        type_colors = {
                            "CONFLICTS_WITH": "#ff4444",
                            "REQUIRES": "#ff8800",
                            "MITIGATES": "#44aa44",
                            "DEPENDS_ON": "#4488ff",
                            "RESTRICTS": "#888888",
                            "MODIFIES": "#aa44ff",
                        }

                        # Draw edges
                        edge_traces = []
                        for e in edges:
                            src = e["source_label"]
                            tgt = e["target_label"]
                            if src in pos and tgt in pos:
                                x0, y0 = pos[src]
                                x1, y1 = pos[tgt]
                                color = type_colors.get(e["dependency_type"], "#cccccc")
                                edge_traces.append(go.Scatter(
                                    x=[x0, x1, None], y=[y0, y1, None],
                                    mode="lines",
                                    line=dict(width=max(1, e.get("strength", 0.5) * 3), color=color),
                                    hoverinfo="text",
                                    text=f"{src} → {tgt}<br>{e['dependency_type']}<br>{e.get('reason', '')}",
                                    showlegend=False,
                                ))

                        # Draw nodes
                        centrality = nx.degree_centrality(G)
                        node_x = [pos[n["label"]][0] for n in nodes if n["label"] in pos]
                        node_y = [pos[n["label"]][1] for n in nodes if n["label"] in pos]
                        node_text = [n["label"] for n in nodes if n["label"] in pos]
                        node_size = [max(15, centrality.get(n["label"], 0) * 80) for n in nodes if n["label"] in pos]

                        # Color by category
                        cat_colors = {
                            "general_information": "#667eea",
                            "restrictive_covenants": "#f56565",
                            "revenue_risks": "#ed8936",
                            "intellectual_property": "#48bb78",
                            "special_provisions": "#9f7aea",
                        }
                        node_color = [cat_colors.get(n.get("category", ""), "#999") for n in nodes if n["label"] in pos]

                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode="markers+text",
                            text=node_text,
                            textposition="top center",
                            textfont=dict(size=9),
                            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
                            hoverinfo="text",
                            hovertext=[
                                f"{n['label']}<br>Category: {n.get('category', 'N/A')}<br>"
                                f"Confidence: {n.get('confidence', 0):.0%}<br>"
                                f"Centrality: {centrality.get(n['label'], 0):.2f}"
                                for n in nodes if n["label"] in pos
                            ],
                        )

                        fig = go.Figure(data=edge_traces + [node_trace])
                        fig.update_layout(
                            height=600,
                            showlegend=False,
                            hovermode="closest",
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            margin=dict(l=0, r=0, t=0, b=0),
                        )

                        # Legend for edge types
                        legend_html = " ".join(
                            f'<span style="color:{c}; font-weight:bold;">■</span> {t}'
                            for t, c in type_colors.items()
                        )
                        st.markdown(f"<small>{legend_html}</small>", unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No dependencies detected in this contract.")

                    # Tabs
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Contradictions", "Missing Requirements", "Impact Rankings", "All Dependencies"
                    ])

                    with tab1:
                        contradictions = api_request("GET", f"/contracts/{selected_id}/contradictions")
                        if contradictions and contradictions.get("contradictions"):
                            for c in contradictions["contradictions"]:
                                with st.expander(f"{c['clause_a']}  vs  {c['clause_b']}", expanded=False):
                                    st.write(f"**Reason:** {c.get('reason', 'N/A')}")
                                    st.write(f"**Strength:** {c.get('strength', 0):.0%}")
                        else:
                            st.success("No contradictions found.")

                    with tab2:
                        completeness = api_request("GET", f"/contracts/{selected_id}/completeness")
                        if completeness and completeness.get("missing_requirements"):
                            for m in completeness["missing_requirements"]:
                                severity = m.get("severity", "MEDIUM")
                                color = "#ff4444" if severity == "HIGH" else "#ff8800"
                                st.markdown(
                                    f'<span style="background:{color};color:white;padding:2px 8px;'
                                    f'border-radius:4px;font-size:0.8em;">{severity}</span> '
                                    f'**{m["missing_label"]}** required by {m["required_by"]}',
                                    unsafe_allow_html=True,
                                )
                                st.write(f"  {m.get('impact', '')}")
                                st.markdown("---")
                        else:
                            st.success("All required clauses are present.")

                    with tab3:
                        rankings = deps.get("recommendations", [])
                        impact = []
                        for e in edges:
                            src = e["source_label"]
                            found = next((i for i in impact if i["label"] == src), None)
                            if found:
                                found["count"] += 1
                            else:
                                impact.append({"label": src, "count": 1})
                        impact.sort(key=lambda x: x["count"], reverse=True)

                        if impact:
                            df = pd.DataFrame(impact[:15])
                            fig = px.bar(
                                df, x="label", y="count",
                                title="Outgoing Dependencies by Clause",
                                labels={"label": "Clause", "count": "Dependencies"},
                            )
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)

                        # Impact explorer
                        st.subheader("Impact Explorer")
                        if nodes:
                            explore_label = st.selectbox(
                                "Select clause to explore impact",
                                [n["label"] for n in nodes],
                            )
                            max_hops = st.slider("Max hops", 1, 5, 3)
                            impact_data = api_request(
                                "GET",
                                f"/contracts/{selected_id}/impact/{explore_label}?max_hops={max_hops}",
                            )
                            if impact_data and impact_data.get("affected_clauses"):
                                st.write(f"**{impact_data['total_affected']}** clauses affected "
                                         f"(max depth: {impact_data['max_depth']})")
                                for ac in impact_data["affected_clauses"]:
                                    st.write(f"  - **{ac['label']}** (depth {ac['depth']}, "
                                             f"via {ac['via']}, {ac['dependency_type']})")
                            else:
                                st.info("No downstream impact detected.")

                    with tab4:
                        if edges:
                            # Type filter
                            all_types = list(set(e["dependency_type"] for e in edges))
                            selected_types = st.multiselect(
                                "Filter by type", all_types, default=all_types
                            )
                            filtered = [e for e in edges if e["dependency_type"] in selected_types]

                            df = pd.DataFrame(filtered)
                            display_cols = ["source_label", "target_label", "dependency_type",
                                            "strength", "reason"]
                            available = [c for c in display_cols if c in df.columns]
                            st.dataframe(df[available], use_container_width=True, hide_index=True)
                        else:
                            st.info("No dependencies to display.")

                    # Recommendations
                    recs = deps.get("recommendations", [])
                    if recs:
                        st.subheader("Recommendations")
                        for i, rec in enumerate(recs, 1):
                            st.write(f"{i}. {rec}")
    else:
        st.warning("Cannot connect to API server.")


elif page == "Search & QA":
    st.header("Search & QA")
    st.markdown("Search the knowledge graph and ask questions about your contracts.")

    tab1, tab2 = st.tabs(["Search", "Ask a Question"])

    with tab1:
        search_query = st.text_input("Search entities and relationships", placeholder="e.g., liability cap, license grant")

        if search_query:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Entities")
                entity_results = api_request("POST", "/search/entities", json={"query": search_query, "limit": 10})
                if entity_results and entity_results.get("results"):
                    for r in entity_results["results"]:
                        p = r["payload"]
                        st.markdown(f"**{p.get('name', '')}** ({p.get('entity_type', '')}) — score: {r['score']:.3f}")
                else:
                    st.info("No entity results. Upload and analyze contracts first.")

            with col2:
                st.subheader("Relationships")
                triple_results = api_request("POST", "/search/triples", json={"query": search_query, "limit": 10})
                if triple_results and triple_results.get("results"):
                    for r in triple_results["results"]:
                        p = r["payload"]
                        pred = p.get('predicate', '').replace('_', ' ').lower()
                        st.markdown(f"**{p.get('subject', '')}** {pred} **{p.get('object', '')}** — score: {r['score']:.3f}")
                else:
                    st.info("No relationship results.")

    with tab2:
        st.subheader("Ask a Question")
        suggested = [
            "Who are the parties to this contract?",
            "What IP rights are licensed?",
            "What are the key obligations?",
            "What are the liability caps?",
            "What is the governing law?",
        ]

        selected_suggestion = st.selectbox("Suggested questions", ["(Type your own)"] + suggested)

        if selected_suggestion != "(Type your own)":
            question = selected_suggestion
        else:
            question = st.text_input("Your question", placeholder="Ask anything about the contracts...")

        if question and st.button("Get Answer"):
            with st.spinner("Retrieving context and generating answer..."):
                result = api_request("POST", "/query", json={"question": question})

            if result:
                confidence = result.get("confidence", 0)
                conf_color = "green" if confidence >= 0.7 else "orange" if confidence >= 0.4 else "red"

                st.markdown(f"**Confidence:** :{conf_color}[{confidence:.0%}]")
                st.markdown("### Answer")
                st.write(result.get("answer", "No answer available"))

                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"Sources ({len(sources)})"):
                        for s in sources:
                            pred = s.get('predicate', '').replace('_', ' ').lower()
                            st.markdown(f"- {s.get('subject', '')} {pred} {s.get('object', '')}")
            else:
                st.error("Failed to get answer. Ensure the API is running and contracts are indexed.")


# Footer
st.markdown("---")
st.markdown(
    "<center>KGGEN-CUAD Contract Analyzer | "
    "Built with Streamlit & FastAPI</center>",
    unsafe_allow_html=True
)
