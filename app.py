"""
AI Visibility Audit - Streamlit Web App
Supports: Single URL, Bulk URL auditing, Multiple LLMs
"""

import streamlit as st
import pandas as pd
import json
import os
import requests
import dspy
from urllib.parse import urlparse
from datetime import datetime
from typing import Dict, List, Optional

# ── DSPy modules (inlined from ai_visibility_audit.py) ───────────────────────


class GenerateSyntheticQueriesCoT(dspy.Signature):
    """
    Given an entity (typically the main topic of a webpage), the current date,
    and a desired number of sub-queries, generate synthetic queries that Google's
    AI search might use internally to understand the entity comprehensively.
    """
    entity: str = dspy.InputField(desc="The main entity/topic to generate queries for")
    current_date: str = dspy.InputField(desc="The current date for context")
    num_queries: str = dspy.InputField(desc="The number of synthetic queries to generate")
    reasoning: str = dspy.OutputField(desc="Brief reasoning about key facets and related concepts")
    synthetic_queries: str = dspy.OutputField(desc="A list of synthetic sub-queries, one per line")


class FanOutQueryCheckerSig(dspy.Signature):
    """Check if given content effectively answers a specific synthetic query."""
    query: str = dspy.InputField(desc="The synthetic query to check")
    content: str = dspy.InputField(desc="The webpage content to check against")
    reasoning: str = dspy.OutputField(desc="Reasoning about content relevance")
    relevance_score: str = dspy.OutputField(desc="Score from 0 to 1 indicating how well content answers the query")


class SyntheticQueryGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_queries = dspy.ChainOfThought(GenerateSyntheticQueriesCoT)

    def forward(self, entity: str, current_date: str, num_queries: int = 10):
        result = self.generate_queries(
            entity=entity,
            current_date=current_date,
            num_queries=str(num_queries),
        )
        queries = [q.strip() for q in result.synthetic_queries.split("\n") if q.strip()]
        return {"reasoning": result.reasoning, "queries": queries[:num_queries]}


class ContentCoverageChecker(dspy.Module):
    def __init__(self):
        super().__init__()
        self.check_relevance = dspy.ChainOfThought(FanOutQueryCheckerSig)

    def forward(self, query: str, content: str):
        result = self.check_relevance(query=query, content=content)
        try:
            score = float(result.relevance_score)
        except Exception:
            score = 0.5
        return {"reasoning": result.reasoning, "relevance_score": min(max(score, 0), 1)}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Visibility Audit",
    page_icon="🔎",
    layout="wide",
)

# ── LLM options ───────────────────────────────────────────────────────────────
LLM_OPTIONS = {
    "Gemini 2.5 Flash (Google)": {
        "model": "gemini/gemini-2.5-flash",
        "api_key_label": "Google API Key",
        "api_key_env": "GOOGLE_API_KEY",
    },
    "GPT-4o (OpenAI)": {
        "model": "gpt-4o",
        "api_key_label": "OpenAI API Key",
        "api_key_env": "OPENAI_API_KEY",
    },
    "Claude 3.5 Sonnet (Anthropic)": {
        "model": "anthropic/claude-3-5-sonnet-20241022",
        "api_key_label": "Anthropic API Key",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
}

# ── DSPy signature for entity extraction (LLM-agnostic) ──────────────────────
class ExtractPageEntity(dspy.Signature):
    """Extract the main entity/topic and key content areas from a webpage."""
    url: str = dspy.InputField(desc="The URL being analyzed")
    content: str = dspy.InputField(desc="Webpage text content (first 2000 chars)")
    entity: str = dspy.OutputField(desc="Main entity or topic in one concise phrase")
    content_chunks: str = dspy.OutputField(desc="3 to 5 key content topics, one per line")


# ── Core helpers ──────────────────────────────────────────────────────────────
def make_lm(llm_choice: str, api_key: str) -> dspy.LM:
    """Create a DSPy LM instance for the selected model."""
    model = LLM_OPTIONS[llm_choice]["model"]
    return dspy.LM(model=model, api_key=api_key, max_tokens=2048)


def fetch_url_content(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AIVisibilityAudit/1.0)"}
    response = requests.get(url, timeout=15, headers=headers)
    response.raise_for_status()
    return response.text


def extract_entity_with_llm(url: str, content: str) -> Dict:
    """Use the configured DSPy LLM to extract entity + content chunks."""
    extractor = dspy.ChainOfThought(ExtractPageEntity)
    result = extractor(url=url, content=content[:2000])
    chunks = [c.strip() for c in result.content_chunks.split("\n") if c.strip()]
    return {"entity": result.entity, "content_chunks": chunks}


def run_audit(
    url: str,
    num_queries: int,
    threshold: float,
    lm: dspy.LM,
    log=None,
) -> Dict:
    """
    Run a full AI visibility audit for one URL.
    lm: DSPy LM instance (thread-safe via dspy.context)
    log: optional callable(str) for streaming status updates.
    """

    def _log(msg: str):
        if log:
            log(msg)

    # Validate URL
    parsed = urlparse(url)
    if not all([parsed.scheme, parsed.netloc]):
        return {"error": True, "message": f"Invalid URL: {url}"}

    # Step 1 – Fetch content
    _log(f"Fetching content from {url} ...")
    try:
        content = fetch_url_content(url)
    except Exception as e:
        return {"error": True, "message": f"Could not fetch URL: {e}"}

    with dspy.context(lm=lm):
        # Step 2 – Extract entity
        _log("Extracting main entity ...")
        try:
            insights = extract_entity_with_llm(url, content)
            entity = insights["entity"]
        except Exception as e:
            return {"error": True, "message": f"Entity extraction failed: {e}"}

        _log(f"Entity identified: **{entity}**")

        # Step 3 – Generate synthetic queries
        _log(f"Generating {num_queries} synthetic queries ...")
        generator = SyntheticQueryGenerator()
        current_date = datetime.now().strftime("%B %d, %Y")
        try:
            query_result = generator.forward(
                entity=entity,
                current_date=current_date,
                num_queries=num_queries,
            )
            synthetic_queries = query_result["queries"]
        except Exception as e:
            return {"error": True, "message": f"Query generation failed: {e}"}

        # Step 4 – Check coverage per query
        checker = ContentCoverageChecker()
        content_for_check = content[:5000]
        coverage_results = []
        covered_count = 0

        for i, query in enumerate(synthetic_queries, 1):
            _log(f"Checking query {i}/{len(synthetic_queries)}: {query}")
            try:
                check = checker.forward(query=query, content=content_for_check)
                score = check["relevance_score"]
                is_covered = score >= threshold
                if is_covered:
                    covered_count += 1
                coverage_results.append({
                    "Query": query,
                    "Score": round(score, 2),
                    "Covered": "✅" if is_covered else "❌",
                    "Reasoning": check["reasoning"],
                })
            except Exception as e:
                coverage_results.append({
                    "Query": query,
                    "Score": 0.0,
                    "Covered": "❌",
                    "Reasoning": f"Error: {e}",
                })

    total = len(synthetic_queries)
    coverage_pct = round((covered_count / total * 100) if total else 0, 1)
    passed = coverage_pct >= (threshold * 100)

    return {
        "url": url,
        "entity": entity,
        "total_queries": total,
        "covered_count": covered_count,
        "coverage_percentage": coverage_pct,
        "threshold": threshold,
        "passed": passed,
        "query_results": coverage_results,
        "error": False,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    llm_choice = st.selectbox("LLM Model", list(LLM_OPTIONS.keys()))
    cfg = LLM_OPTIONS[llm_choice]

    default_key = os.getenv(cfg["api_key_env"], "")
    api_key = st.text_input(cfg["api_key_label"], value=default_key, type="password")

    st.divider()

    num_queries = st.slider("Synthetic Queries", min_value=3, max_value=15, value=10)
    threshold = st.slider(
        "Coverage Threshold",
        min_value=0.30, max_value=0.90,
        value=0.55, step=0.05,
        help="Minimum relevance score to count a query as covered",
    )

    st.divider()
    st.caption("API keys are never stored or logged.")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔎 AI Visibility Audit")
st.caption(
    "Simulate Google's query fan-out process to measure how well your content "
    "would be discovered by AI search."
)

tab_single, tab_bulk = st.tabs(["Single URL", "Bulk Audit"])


# ── TAB 1 – Single URL ────────────────────────────────────────────────────────
with tab_single:
    url_input = st.text_input(
        "URL to audit",
        placeholder="https://example.com/your-page",
    )

    if st.button("Run Audit", type="primary", key="btn_single"):
        if not api_key:
            st.error(f"Enter your {cfg['api_key_label']} in the sidebar.")
        elif not url_input.strip():
            st.error("Enter a URL above.")
        else:
            lm = make_lm(llm_choice, api_key)

            messages: List[str] = []
            status_box = st.empty()

            def log_single(msg: str):
                messages.append(msg)
                status_box.info("\n\n".join(messages))

            with st.spinner("Running audit ..."):
                result = run_audit(url_input.strip(), num_queries, threshold, lm=lm, log=log_single)

            status_box.empty()

            if result.get("error"):
                st.error(f"Audit failed: {result['message']}")
            else:
                # Metrics row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Coverage", f"{result['coverage_percentage']}%")
                c2.metric("Covered Queries", f"{result['covered_count']} / {result['total_queries']}")
                c3.metric("Entity", result["entity"])
                c4.metric("Result", "✅ PASSED" if result["passed"] else "❌ FAILED")

                st.divider()

                # Query breakdown table
                st.subheader("Query Breakdown")
                df = pd.DataFrame(result["query_results"])
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Downloads
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(result, indent=2),
                        file_name=f"audit_{urlparse(url_input).netloc}.json",
                        mime="application/json",
                    )
                with col_dl2:
                    csv_single = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        data=csv_single,
                        file_name=f"audit_{urlparse(url_input).netloc}.csv",
                        mime="text/csv",
                    )


# ── TAB 2 – Bulk Audit ────────────────────────────────────────────────────────
with tab_bulk:
    st.markdown(
        "Upload a CSV with one URL per row. "
        "The first column is used as the URL list (header optional)."
    )

    # Sample download
    st.download_button(
        "Download sample CSV",
        data="url\nhttps://example.com/page1\nhttps://example.com/page2\n",
        file_name="sample_urls.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if st.button("Run Bulk Audit", type="primary", key="btn_bulk"):
        if not api_key:
            st.error(f"Enter your {cfg['api_key_label']} in the sidebar.")
        elif uploaded_file is None:
            st.error("Upload a CSV file first.")
        else:
            df_input = pd.read_csv(uploaded_file)
            urls = df_input.iloc[:, 0].dropna().astype(str).str.strip().tolist()
            urls = [u for u in urls if u and u.lower() != "url"]  # strip header if included

            if not urls:
                st.error("No valid URLs found in the uploaded file.")
            else:
                lm = make_lm(llm_choice, api_key)

                st.info(f"Auditing {len(urls)} URL(s) — this may take a few minutes ...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                summary_rows = []
                full_results = []

                for i, url in enumerate(urls):
                    status_text.text(f"[{i+1}/{len(urls)}] Auditing: {url}")
                    result = run_audit(url, num_queries, threshold, lm=lm)
                    full_results.append(result)

                    if result.get("error"):
                        summary_rows.append({
                            "URL": url,
                            "Entity": "—",
                            "Coverage %": "—",
                            "Covered": "—",
                            "Total Queries": "—",
                            "Passed": "❌",
                            "Error": result["message"],
                        })
                    else:
                        summary_rows.append({
                            "URL": url,
                            "Entity": result["entity"],
                            "Coverage %": result["coverage_percentage"],
                            "Covered": result["covered_count"],
                            "Total Queries": result["total_queries"],
                            "Passed": "✅" if result["passed"] else "❌",
                            "Error": "",
                        })

                    progress_bar.progress((i + 1) / len(urls))

                status_text.empty()
                progress_bar.empty()

                df_summary = pd.DataFrame(summary_rows)

                st.subheader("Bulk Results Summary")
                st.dataframe(df_summary, use_container_width=True, hide_index=True)

                # Summary stats
                total_audited = len(urls)
                passed_count = sum(1 for r in full_results if not r.get("error") and r["passed"])
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Audited", total_audited)
                m2.metric("Passed", passed_count)
                m3.metric("Failed / Error", total_audited - passed_count)

                st.divider()

                # Downloads
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.download_button(
                        "Download Summary CSV",
                        data=df_summary.to_csv(index=False),
                        file_name=f"bulk_audit_{timestamp}.csv",
                        mime="text/csv",
                    )
                with col_b2:
                    st.download_button(
                        "Download Full JSON",
                        data=json.dumps(full_results, indent=2),
                        file_name=f"bulk_audit_{timestamp}.json",
                        mime="application/json",
                    )
