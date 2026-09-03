import base64
import io
import os
import re
import unicodedata
from datetime import datetime
from html import escape
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from career_ai import CareerAI
from career_config import GROQ_API_KEY, TAVILY_API_KEY
from career_documents import classify_document, extract_document
from career_intelligence import ExplainableResumeScorer, QueryRouter, SourceAwareRetriever, DocumentChunk, WebResearcher, career_report_markdown, grounded_prompt
from career_rag import CareerRAG
from career_research import research_context, format_research
from document_ingestion import ingest_document, DocumentIngestionError
from pdf_processor import PDFSummarizer, SummaryFormatter
from utility import TextProcessor, ErrorHandler

load_dotenv()
st.set_page_config(page_title="AI Career Intelligence",
                   page_icon="🚀", layout="wide")


@st.cache_data(show_spinner=False)
def pdf_safe_text(value: object) -> str:
    replacements = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"',
                                 "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "})
    return unicodedata.normalize("NFKD", str(value)).translate(replacements).encode("ascii", "replace").decode("ascii")


def extract_pdf_text(reader: PdfReader, pdf_bytes: bytes):
    page_text = [page.extract_text() or "" for page in reader.pages]
    if all(t.strip() for t in page_text):
        return "\n".join(page_text).strip(), False
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError as e:
        raise RuntimeError(
            "OCR requires pypdfium2, pytesseract and the Tesseract binary.") from e
    doc = pdfium.PdfDocument(pdf_bytes)
    out = []
    for i, t in enumerate(page_text):
        if t.strip():
            out.append(t)
            continue
        page = doc[i]
        out.append(pytesseract.image_to_string(
            page.render(scale=2.2).to_pil()))
        page.close()
    doc.close()
    return "\n".join(out).strip(), True


def extract_numeric_statistics(text):
    rows = []
    pattern = re.compile(
        r"^\s*([A-Za-z][A-Za-z &'()/.-]{1,40})\s*(?::|[-–])\s*(?:[$€£₹]\s*)?(-?\d[\d,]*(?:\.\d+)?)\s*(%|percent)?\s*$", re.I)
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if m:
            rows.append({"Statistic": m.group(1).strip().title(), "Value": float(
                m.group(2).replace(",", "")), "Unit": "%" if m.group(3) else ""})
    return pd.DataFrame(rows).drop_duplicates(subset=["Statistic"]) if rows else pd.DataFrame(columns=["Statistic", "Value", "Unit"])


def create_summary_pdf(filename, summary, stats, numeric):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=.65*inch,
                            leftMargin=.65*inch, topMargin=.6*inch, bottomMargin=.6*inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontSize=20,
                           alignment=TA_CENTER, textColor=colors.HexColor("#312e81"))
    body = ParagraphStyle(
        "b", parent=styles["BodyText"], fontSize=10, leading=15)
    story = [Paragraph("Executive PDF Summary", title), Paragraph(f"{escape(pdf_safe_text(filename))} | {datetime.now():%B %d, %Y}", body), Spacer(1, 12), Paragraph("Document Overview", styles["Heading2"]), Table([["Pages", "Words", "Characters", "Reading time"], [stats.get("pages", "-"), f"{stats.get('words', 0):,}", f"{stats.get('characters', 0):,}", stats.get(
        "reading_time", "-")]], style=TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312e81")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)])), Paragraph("Summary", styles["Heading2"]), Paragraph(escape(pdf_safe_text(summary)).replace("\n", "<br/>"), body)]
    if not numeric.empty:
        story += [Paragraph("Extracted Statistics", styles["Heading2"]), Table([["Statistic", "Value", "Unit"]]+[[r["Statistic"], f"{r['Value']:,.2f}".rstrip("0").rstrip("."), r["Unit"]] for _, r in numeric.iterrows(
        )], style=TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312e81")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]))]
    doc.build(story)
    return buf.getvalue()


st.markdown("<style>.main .block-container{max-width:1400px;padding-top:1.5rem}.hero{padding:2rem 2.5rem;border-radius:18px;background:linear-gradient(135deg,#0f172a,#312e81);color:white;margin-bottom:1.5rem;text-align:center;}.hero h1{margin:0}.muted{color:#64748b}</style>", unsafe_allow_html=True)
st.markdown("<div class='hero'><h1>🚀 AI Career Intelligence with PDF Summariser</h1><p>Resume + JD intelligence research, explainable matching, RAG, external research and career generation.</p></div>", unsafe_allow_html=True)

if "docs" not in st.session_state:
    st.session_state.docs = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "research" not in st.session_state:
    st.session_state.research = []
if "token_usage" not in st.session_state:
    st.session_state.token_usage = {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}


def record_usage(usage):
    for key in st.session_state.token_usage:
        st.session_state.token_usage[key] += usage.get(key, 0)


with st.sidebar:
    st.header("📂 Career Workspace")
    company = st.text_input("Company name", placeholder="e.g. Microsoft",
                            help="Used to focus interview preparation and web research.")
    files = st.file_uploader("Resume, JD and supporting documents", type=[
                             "pdf", "docx", "txt", "md", "csv", "json", "log"], accept_multiple_files=True)
    web_enabled = st.checkbox("Enable external research", value=bool(
        TAVILY_API_KEY), disabled=not bool(TAVILY_API_KEY))
    if not TAVILY_API_KEY:
        st.caption("Adding web search option consumes a lot of tokens")

    st.subheader("Token usage")
    usage = st.session_state.token_usage
    u1, u2 = st.columns(2)
    u1.metric("Total tokens", f"{usage['total_tokens']:,}")
    u2.metric("Requests", f"{usage['requests']:,}")
    st.caption(
        f"Input: {usage['prompt_tokens']:,} • Output: {usage['completion_tokens']:,}")
    st.caption("Designed by Kam Rachakonda")
    if files and st.button("📥 Index documents", use_container_width=True):
        docs = []
        for f in files:
            try:
                text, meta = extract_document(f.name, f.getvalue())
                docs.append({"filename": f.name, "text": text, "type": classify_document(
                    f.name, text), "meta": meta})
            except Exception as e:
                st.error(f"{f.name}: {e}")
        if docs:
            rag = CareerRAG()
            rag.add_documents(docs)
            st.session_state.docs = docs
            st.session_state.rag = rag
            st.success(f"Indexed {len(docs)} document(s)")
    if st.session_state.docs:
        st.subheader("Indexed")
        [st.write(f"**{d['type']}** — {d['filename']}")
         for d in st.session_state.docs]

st.markdown("## 1. Career Intelligence")
if not st.session_state.docs:
    st.info("Upload and index your resume and target JD above to unlock career intelligence. You can also use the PDF Summariser below independently.")
else:
    resumes = [d for d in st.session_state.docs if d["type"] == "Resume"]
    pds = [d for d in st.session_state.docs if d["type"]
           == "Position Description"]
    names = [d["filename"] for d in st.session_state.docs]
    c1, c2 = st.columns(2)
    with c1:
        rn = st.selectbox("Resume", [d["filename"]
                          for d in resumes] or names, key="resume_choice")
    with c2:
        pn = st.selectbox(
            "Target JD / PD", [d["filename"] for d in pds] or names, key="pd_choice")
    resume_text = next(d["text"]
                       for d in st.session_state.docs if d["filename"] == rn)
    jd_text = next(d["text"]
                   for d in st.session_state.docs if d["filename"] == pn)
    scorer = ExplainableResumeScorer()
    score = scorer.score(resume_text, jd_text)
    m = st.columns(4)
    m[0].metric("Overall Fit", f"{score.overall:.1f}/100")
    m[1].metric("Resume words", f"{len(resume_text.split()):,}")
    m[2].metric("JD words", f"{len(jd_text.split()):,}")
    m[3].metric("Research", len(st.session_state.research))
    st.dataframe(pd.DataFrame([{"Dimension": d.name, "Score": d.score, "Weight": f"{scorer.WEIGHTS[d.name]:.0%}", "Matched": ", ".join(
        d.matched[:6]), "Gaps": ", ".join(d.missing[:6])} for d in score.dimensions]), use_container_width=True, hide_index=True)
    tabs = st.tabs(["🎯 Gaps & Evidence", "🤖 Advisor", "✍️ Resume / LinkedIn",
                   "✉️ Cover Letter", "🎤 Interview", "🌐 Research", "📥 Report"])
    with tabs[0]:
        for d in score.dimensions:
            with st.expander(f"{d.name} — {d.score:.1f}/100"):
                st.write("**Matched:**", ", ".join(d.matched) or "None")
                st.write("**Missing:**", ", ".join(d.missing) or "None")
                st.write("**Evidence:**")
                [st.caption(e) for e in d.evidence]
    with tabs[1]:
        q = st.text_area("Ask the Career Advisor",
                         placeholder="Should I apply? How should I position myself? What gaps matter most?", key="advisor_q")
        if st.button("Ask Advisor", key="advisor_btn") and q:
            route = QueryRouter().route(q, True, web_enabled)
            context = st.session_state.rag.context(
                q, 8) if st.session_state.rag else ""
            research = ""
            if route.value in {"web", "hybrid"} and web_enabled:
                queries = [q]
                if company:
                    queries.append(f"{company} hiring priorities skills 2026")
                st.session_state.research = research_context(queries)
                research = format_research(st.session_state.research)
            try:
                st.markdown(CareerAI(usage_callback=record_usage).advisor(
                    q, context, research))
                st.caption(
                    f"Source route: {route.value} — {QueryRouter().explain(q, route)}")
            except Exception as e:
                st.error(str(e))
    with tabs[2]:
        task = st.text_area(
            "Optimisation goal", value="Create a stronger executive summary, LinkedIn headline/About and role-relevant experience bullets. Preserve every fact and metric.", key="position_task")
        if st.button("Generate positioning", key="position_btn"):
            try:
                st.markdown(CareerAI(usage_callback=record_usage).generate(task, grounded_prompt(
                    task, [f"Resume: {resume_text}", f"Job description: {jd_text}"])))
            except Exception as e:
                st.error(str(e))
    with tabs[3]:
        if st.button("Generate cover letter", key="cover_btn"):
            try:
                st.markdown(CareerAI(usage_callback=record_usage).generate(
                    "Write a tailored cover letter for this role using only supported candidate evidence. Mark unsupported facts [NEEDS CONFIRMATION].", f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd_text}"))
            except Exception as e:
                st.error(str(e))
    with tabs[4]:
        if st.button("Generate interview pack", key="interview_btn"):
            try:
                source = f"RESUME:\n{resume_text}\n\nROLE:\n{jd_text}" + \
                    (f"\n\nCOMPANY:\n{company}" if company else "")
                st.markdown(CareerAI(usage_callback=record_usage).generate(
                    "Create 10 likely interview questions, answer frameworks and STAR story prompts mapped to this role. Do not invent candidate stories.", source))
            except Exception as e:
                st.error(str(e))
    with tabs[5]:
        urls = st.text_area(
            "Public research URLs (one per line)", key="research_urls")
        if st.button("Research sources", key="research_btn"):
            st.session_state.research = []
            for url in urls.splitlines():
                if url.strip():
                    try:
                        st.session_state.research.append(
                            WebResearcher().fetch(url.strip()))
                    except Exception as e:
                        st.warning(str(e))
        for e in st.session_state.research:
            st.markdown(f"**{e.citation}**\n\n{e.text[:1500]}")
    with tabs[6]:
        recs = [
            f"Strengthen evidence for {d.name}: {', '.join(d.missing) or 'maintain current evidence'}." for d in score.dimensions]
        report = career_report_markdown(score, recs, st.session_state.research)
        st.download_button("📥 Download Career Intelligence Report",
                           report, "career_intelligence_report.md", "text/markdown")
        st.download_button("📄 Download Report as Text", report,
                           "career_intelligence_report.txt", "text/plain")

st.divider()
st.markdown("## 2. AI PDF Summariser")
st.caption("The original OCR, document statistics, charts, section summaries, key quotes and executive PDF export are available.")
uploaded = st.file_uploader("Upload a PDF to summarize", type=[
                            "pdf"], key="pdf_summary")
if uploaded:
    try:
        reader = PdfReader(uploaded)
        text, ocr = extract_pdf_text(reader, uploaded.getvalue())
        stats = TextProcessor.get_text_statistics(text)
        numeric = extract_numeric_statistics(text)
        if ocr:
            st.info("Scanned pages detected — OCR was used.")
        c = st.columns(4)
        c[0].metric("Pages", len(reader.pages))
        c[1].metric("Words", f"{stats['words']:,}")
        c[2].metric("Characters", f"{stats['characters']:,}")
        c[3].metric("Read time", stats['reading_time'])
        summary_type = st.selectbox("Summary format", [
                                    "concise", "detailed", "bullet_points", "executive", "bar_chart", "pie_chart"])
        custom = st.text_area(
            "Custom summary instruction", key="summary_prompt")
        if summary_type in {"bar_chart", "pie_chart"} and len(numeric) >= 2:
            st.dataframe(numeric, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(numeric, x="Statistic", y="Value", text_auto=True) if summary_type ==
                            "bar_chart" else px.pie(numeric, names="Statistic", values="Value"), use_container_width=True)
        if st.button("🚀 Generate PDF Summary", key="pdf_btn"):
            summarizer = PDFSummarizer()
            result = summarizer.summarize_chunks(
                summarizer.chunk_text(text), summary_type, custom)
            record_usage(summarizer.usage)
            formatted = SummaryFormatter.format_summary(
                result["combined_summary"])
            st.markdown(formatted)
            st.download_button("Download summary text", formatted,
                               f"summary_{pdf_safe_text(uploaded.name)}.txt", "text/plain")
            st.download_button("Download executive PDF", create_summary_pdf(uploaded.name, formatted, {
                               **stats, "pages": len(reader.pages)}, numeric), f"executive_summary_{pdf_safe_text(uploaded.name)}.pdf", "application/pdf")
            with st.expander("Section summaries"):
                [st.markdown(f"**Section {x['chunk_number']}**\n\n{x['summary']}")
                 for x in result["individual_summaries"]]
            if st.checkbox("Show document structure analysis", value=True):
                st.markdown(summarizer.analyze_document_structure(
                    text).get("analysis", ""))
            if st.checkbox("Show key quotes", value=True):
                [st.markdown(f"> {q}")
                 for q in summarizer.extract_key_quotes(text)]
    except Exception as e:
        st.error(ErrorHandler.handle_error(e, context="PDF Processing"))

st.markdown("<p style='text-align:center;font-size:1.15rem;font-weight:600'>AI Career Intelligence • Streamlit + Groq + semantic RAG + Tavily external research • AI PDF Summariser</p>", unsafe_allow_html=True)
