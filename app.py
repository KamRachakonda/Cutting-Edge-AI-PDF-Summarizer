import os
import re
import io
import base64
from datetime import datetime
from html import escape
import unicodedata
import streamlit as st
import pandas as pd
import plotly.express as px
from pypdf import PdfReader
from dotenv import load_dotenv

# Import backend classes
from pdf_processor import PDFSummarizer, SummaryFormatter
from utility import TextProcessor, ErrorHandler
from career_intelligence import (
    ExplainableResumeScorer,
    QueryRouter,
    SourceAwareRetriever,
    WebResearcher,
    career_report_markdown,
    grounded_prompt,
)
from document_ingestion import ingest_document, DocumentIngestionError


with open("pdf-summarizer-banner.png", "rb") as banner_file:
    banner_image = base64.b64encode(banner_file.read()).decode("ascii")


def pdf_safe_text(value: object) -> str:
    """Convert generated text to characters supported by ReportLab's built-in fonts."""
    replacements = str.maketrans({
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    })
    normalized = unicodedata.normalize(
        "NFKD", str(value)).translate(replacements)
    return normalized.encode("ascii", "replace").decode("ascii")


def extract_pdf_text(reader: PdfReader, pdf_bytes: bytes) -> tuple[str, bool]:
    """Extract embedded text and OCR pages that contain scanned images."""
    page_text = [page.extract_text() or "" for page in reader.pages]
    if all(text.strip() for text in page_text):
        return "\n".join(page_text).strip(), False

    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError as error:
        raise RuntimeError(
            "OCR support is unavailable. Install the project requirements and Tesseract OCR."
        ) from error

    rendered_document = pdfium.PdfDocument(pdf_bytes)
    ocr_text = []
    for page_number, text in enumerate(page_text):
        if text.strip():
            ocr_text.append(text)
            continue
        page = rendered_document[page_number]
        image = page.render(scale=2.2).to_pil()
        ocr_text.append(pytesseract.image_to_string(image))
        page.close()
    rendered_document.close()
    return "\n".join(ocr_text).strip(), True


def extract_numeric_statistics(text: str) -> pd.DataFrame:
    """Extract simple label/value statistics from lines in extracted PDF text."""
    statistics = []
    pattern = re.compile(
        r"^\s*([A-Za-z][A-Za-z &'()/.-]{1,40})\s*(?::|[-–])\s*"
        r"(?:[$€£₹]\s*)?(-?\d[\d,]*(?:\.\d+)?)\s*(%|percent)?\s*$",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            label, value, unit = match.groups()
            statistics.append({
                "Statistic": label.strip().title(),
                "Value": float(value.replace(",", "")),
                "Unit": "%" if unit else "",
            })
    return pd.DataFrame(statistics).drop_duplicates(subset=["Statistic"])


def create_summary_pdf(
    filename: str,
    summary: str,
    document_stats: dict,
    numeric_stats: pd.DataFrame,
) -> bytes:
    """Create a print-ready executive summary PDF in memory."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as error:
        raise RuntimeError(
            "PDF export requires ReportLab. Install it with: pip install reportlab"
        ) from error

    safe_filename = pdf_safe_text(filename)
    pdf_buffer = io.BytesIO()
    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"Executive Summary - {safe_filename}",
        author="Kam Rachakonda",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=27,
        textColor=colors.HexColor("#312e81"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=22,
    )
    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#1e1b4b"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8,
    )

    story = [
        Paragraph("Executive PDF Summary", title_style),
        Paragraph(
            f"{escape(safe_filename)} &nbsp;|&nbsp; Generated {datetime.now().strftime('%B %d, %Y')}",
            subtitle_style,
        ),
        Paragraph("Document Overview", heading_style),
    ]
    overview_data = [
        ["Pages", "Words", "Characters", "Estimated reading time"],
        [
            pdf_safe_text(document_stats.get("pages", "-")),
            pdf_safe_text(f"{document_stats.get('words', 0):,}"),
            pdf_safe_text(f"{document_stats.get('characters', 0):,}"),
            pdf_safe_text(document_stats.get("reading_time", "-")),
        ],
    ]
    overview_table = Table(overview_data, colWidths=[1.65 * inch] * 4)
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312e81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#eef2ff")),
        ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#1e293b")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2fe")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c7d2fe")),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([overview_table, Paragraph("Summary", heading_style)])
    summary_text = escape(pdf_safe_text(summary)).replace("\n", "<br/>")
    story.append(Paragraph(summary_text, body_style))

    if not numeric_stats.empty:
        story.append(Paragraph("Extracted Statistics", heading_style))
        statistics_data = [["Statistic", "Value", "Unit"]]
        for _, row in numeric_stats.iterrows():
            statistics_data.append([
                escape(pdf_safe_text(row["Statistic"])),
                f"{row['Value']:,.2f}".rstrip("0").rstrip("."),
                escape(pdf_safe_text(row["Unit"])),
            ])
        statistics_table = Table(statistics_data, colWidths=[
                                 3.5 * inch, 1.5 * inch, 1 * inch])
        statistics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#312e81")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8fafc")]),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(statistics_table)

    story.extend([
        Spacer(1, 28),
        Paragraph("Designed by Kam Rachakonda", ParagraphStyle(
            "Signature",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
        )),
    ])
    document.build(story)
    return pdf_buffer.getvalue()


# Load environment variables (.env)
load_dotenv()
GITHUB_URL = "https://github.com/KamRachakonda/Cutting-Edge-AI-PDF-Summarizer"

# Page configuration
st.set_page_config(
    page_title="SummarizeAI — Intelligent PDF Studio",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
        /* Global layout tweaks */
        .main .block-container {
        padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1000px;
        }

        /* Hero Header */
        .hero-container {
            min-height: 310px;
            display: flex;
            align-items: center;
            padding: 2.5rem 4rem;
            background: linear-gradient(90deg, rgba(15, 23, 42, .96) 0%, rgba(30, 27, 75, .84) 38%, rgba(49, 46, 129, .18) 78%), url('data:image/png;base64,__BANNER_IMAGE__') center/cover;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 18px 35px -12px rgba(15, 23, 42, .4);
            overflow: hidden;
        }
        .hero-content {
            max-width: 530px;
            text-align: left;
        }
        .hero-kicker {
            display: inline-block;
            margin-bottom: .9rem;
            color: #93c5fd;
            font-size: .76rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .hero-title {
            font-size: 2.65rem;
            line-height: 1.1;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0;
            color: #ffffff;
        }
        .hero-subtitle {
            max-width: 460px;
            font-size: 1.05rem;
            color: #dbeafe;
            margin: 1rem 0 0;
            line-height: 1.65;
            font-weight: 400;
        }

        /* Stat Card Badges */
        .metric-container {
        background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }
        .metric-value {
        font-size: 1.6rem;
            font-weight: 700;
            color: #1e293b;
        }
        .metric-label {
        font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-top: 0.2rem;
        }

        /* Summary Card Box */
        .summary-card {
        background: #ffffff;
            border-radius: 14px;
            padding: 1.8rem;
            border: 1px solid #e2e8f0;
            border-left: 5px solid #4f46e5;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            margin-top: 1.5rem;
            line-height: 1.7;
            font-size: 1rem;
            color: #334155;
        }

        /* Buttons */
        div.stButton > button:first-child {
        width: 100%;
            height: 3rem;
            font-size: 1.05rem;
            font-weight: 600;
            border-radius: 10px;
            background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
            color: white;
            border: none;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.35);
        }
        div.stButton > button:first-child:hover {
        transform: translateY(-1px);
            box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.45);
        }
        @media (max-width: 700px) {
            .hero-container { min-height: 330px; padding: 2rem 1.5rem; background-position: center; }
            .hero-title { font-size: 2.05rem; }
        }
    </style>
    """.replace("__BANNER_IMAGE__", banner_image),
    unsafe_allow_html=True,
)

# Hero Header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-content">
            <div class="hero-kicker">AI-powered document intelligence</div>
            <h1 class="hero-title">✨ Cutting-Edge AI PDF Summarizer</h1>
            <p class="hero-subtitle">Save hours by turning complex papers into instant insights. Stop reading pages. Start getting answers instantly.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    with st.expander("⚙️ Summary Settings", expanded=True):
        st.caption("Customize how your document should be analyzed.")

        summary_type_options = {
            "concise": "⚡ Concise (5-10 Sentences)",
            "detailed": "📑 Detailed (Headings & Depth)",
            "bullet_points": "📌 Key Bullet Points",
            "executive": "💼 Executive (Insights & Actions)",
            "bar_chart": "📊 Bar Chart + Summary",
            "pie_chart": "🥧 Pie Chart + Summary",
        }

        selected_type_key = st.selectbox(
            "Format Style",
            options=list(summary_type_options.keys()),
            format_func=lambda x: summary_type_options[x],
            index=0,
        )

        st.markdown("---")
        st.markdown("### 🎯 Fine-Tune Instructions")
        custom_prompt = st.text_area(
            "Custom Prompt",
            placeholder="e.g., Focus on numerical KPIs, financial results, or technical architecture...",
            height=120,
        )
        include_analysis = st.checkbox(
            "Analyze document structure", value=True)
        include_quotes = st.checkbox("Extract key quotes", value=True)

    st.markdown("---")
    st.caption("⚡ Powered by Advanced AI Processing")
    st.markdown("### Career Intelligence")
    career_query = st.text_input(
        "Question router",
        placeholder="e.g., What skills am I missing for this role?",
    )
    if career_query:
        route = QueryRouter().route(
            career_query,
            has_documents=st.session_state.get("uploaded_file_present", False),
        )
        research_urls = st.text_area(
            "Company or role research URLs",
            placeholder="One public company or role URL per line",
            height=80,
        )
        st.info(
            f"Route: {route.value.title()} | {QueryRouter().explain(career_query, route)}")
    st.markdown(f"[View project on GitHub]({GITHUB_URL})")
    st.caption("Designed by Kam Rachakonda")

# Main Content: File Upload
upload_spacer, upload_column = st.columns([1.35, 1])
with upload_column:
    uploaded_file = st.file_uploader(
        "Upload PDF document(s)",
        type=["pdf"],
        help="Each PDF file must be less than 200 MB.",
    )

st.session_state["uploaded_file_present"] = uploaded_file is not None
if uploaded_file:
    try:
        reader = PdfReader(uploaded_file)
        extracted_text, used_ocr = extract_pdf_text(
            reader, uploaded_file.getvalue())

        if not extracted_text:
            st.warning(
                "⚠️ No readable text detected in this document. It may consist entirely of scanned images."
            )
        else:
            if used_ocr:
                st.info("Scanned pages detected. OCR was used to read the document.")
            stats = TextProcessor.get_text_statistics(extracted_text)
            numeric_stats = extract_numeric_statistics(extracted_text)
            metadata = reader.metadata or {}

            with st.expander("📄 PDF Metadata", expanded=False):
                st.json({
                    "title": metadata.get("/Title", "Unknown"),
                    "author": metadata.get("/Author", "Unknown"),
                    "subject": metadata.get("/Subject", "Unknown"),
                    "pages": len(reader.pages),
                })

            # Metadata Display
            st.markdown("#### 📊 Document Overview")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-value">{len(reader.pages)}</div>
                        <div class="metric-label">Pages</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-value">{stats["words"]:,}</div>
                        <div class="metric-label">Words</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-value">{stats["characters"]:,}</div>
                        <div class="metric-label">Characters</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col4:
                st.markdown(
                    f"""
                    <div class="metric-container">
                        <div class="metric-value">{stats["reading_time"]}</div>
                        <div class="metric-label">Est. Read Time</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")

            if selected_type_key in {"bar_chart", "pie_chart"}:
                st.markdown("#### 📈 Document Statistics")
                if len(numeric_stats) >= 2:
                    st.dataframe(numeric_stats, hide_index=True,
                                 use_container_width=True)
                    with st.container(border=True):
                        if selected_type_key == "bar_chart":
                            figure = px.bar(
                                numeric_stats,
                                x="Statistic",
                                y="Value",
                                color="Statistic",
                                text_auto=True,
                                title="Extracted PDF Statistics",
                            )
                        else:
                            figure = px.pie(
                                numeric_stats,
                                names="Statistic",
                                values="Value",
                                title="Extracted PDF Statistics",
                            )
                        figure.update_layout(
                            showlegend=True, margin=dict(t=70, l=20, r=20, b=20))
                        st.plotly_chart(figure, use_container_width=True)
                else:
                    st.info(
                        "No labeled statistics were detected. Add labels such as 'Revenue: 1200' to create a chart.")

            # Action Button
            if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
                with st.spinner("Analyzing document and extracting insights..."):
                    summarizer = PDFSummarizer()
                    chunks = summarizer.chunk_text(extracted_text)
                    summary_result = summarizer.summarize_chunks(
                        chunks=chunks,
                        summary_type=selected_type_key,
                        custom_prompt=custom_prompt,
                    )
                    raw_summary = summary_result["combined_summary"]
                    formatted_summary = SummaryFormatter.format_summary(
                        raw_summary)

                    st.markdown("### 📝 Generated Summary")
                    st.markdown(
                        f"""
                        <div class="summary-card">
                            {formatted_summary}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.download_button(
                        label="📥 Download Summary as Text",
                        data=formatted_summary,
                        file_name=f"summary_{pdf_safe_text(uploaded_file.name.replace('.pdf', ''))}.txt",
                        mime="text/plain",
                    )

                    pdf_data = create_summary_pdf(
                        filename=uploaded_file.name,
                        summary=formatted_summary,
                        document_stats={
                            **stats,
                            "pages": len(reader.pages),
                        },
                        numeric_stats=numeric_stats,
                    )
                    st.download_button(
                        label="📄 Download Executive Summary as PDF",
                        data=pdf_data,
                        file_name=f"executive_summary_{pdf_safe_text(uploaded_file.name.replace('.pdf', ''))}.pdf",
                        mime="application/pdf",
                    )

                    st.caption(
                        f"Processed {summary_result['total_chunks']} document section(s)."
                    )
                    with st.expander("Section Summaries", expanded=False):
                        for item in summary_result["individual_summaries"]:
                            st.markdown(f"**Section {item['chunk_number']}**")
                            st.markdown(item["summary"])
                    if include_analysis:
                        analysis_result = summarizer.analyze_document_structure(
                            extracted_text)
                        with st.expander("Document Structure Analysis", expanded=True):
                            if analysis_result["status"] == "success":
                                st.markdown(analysis_result["analysis"])
                            else:
                                st.warning(analysis_result["analysis"])
                    if include_quotes:
                        with st.expander("Key Quotes", expanded=False):
                            for quote in summarizer.extract_key_quotes(extracted_text):
                                st.markdown(f"> {quote}")

    except Exception as e:
        error_msg = ErrorHandler.handle_error(e, context="PDF Processing")
        st.error(error_msg)


st.markdown("---")
st.markdown("## Career Intelligence Workspace")
st.caption("Evidence-backed resume analysis and career document generation.")
career_col1, career_col2 = st.columns(2)
with career_col1:
    resume_text = st.text_area(
        "Resume text",
        placeholder="Paste the resume text to score and optimize.",
        height=220,
    )
with career_col2:
    jd_text = st.text_area(
        "Job description",
        placeholder="Paste the target job description.",
        height=220,
    )

docx_file = st.file_uploader(
    "Optional DOCX resume or job description",
    type=["docx"],
    key="career_docx",
)
if docx_file:
    try:
        docx_chunks = ingest_document(docx_file.getvalue(), docx_file.name)
        docx_text = "\n\n".join(chunk.text for chunk in docx_chunks)
        st.success(
            f"Extracted {len(docx_text.split()):,} words from {docx_file.name}.")
        st.text_area("Extracted DOCX text", docx_text,
                     height=150, disabled=True)
    except DocumentIngestionError as error:
        st.error(str(error))

if resume_text and jd_text:
    scorer = ExplainableResumeScorer()
    score = scorer.score(resume_text, jd_text)
    st.metric("Explainable resume match", f"{score.overall:.2f}/100")
    score_columns = st.columns(len(score.dimensions))
    for column, dimension in zip(score_columns, score.dimensions):
        with column:
            st.metric(dimension.name.title(), f"{dimension.score:.2f}/100")
            if dimension.matched:
                st.caption(f"Matched: {', '.join(dimension.matched[:6])}")
            if dimension.missing:
                st.caption(f"Missing: {', '.join(dimension.missing[:6])}")

    recommendations = [
        f"Address missing {dimension.name} terms only with truthful evidence: "
        f"{', '.join(dimension.missing) or 'none'}"
        for dimension in score.dimensions
    ]
    report = career_report_markdown(score, recommendations)
    research_evidence = []
    if research_urls:
        for url in research_urls.splitlines():
            if not url.strip():
                continue
            try:
                research_evidence.append(WebResearcher().fetch(url))
            except Exception as error:
                st.warning(f"Research source skipped: {error}")
        if research_evidence:
            st.info(
                f"Loaded {len(research_evidence)} cited research source(s).")
            report = career_report_markdown(
                score, recommendations, research_evidence)
    st.download_button(
        "Download career intelligence report",
        data=report,
        file_name="career_intelligence_report.md",
        mime="text/markdown",
    )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Optimize LinkedIn profile", use_container_width=True):
            try:
                summarizer = PDFSummarizer()
                evidence = [
                    f"Resume: {line}" for line in resume_text.splitlines() if line.strip()]
                result = summarizer.summarize(
                    [grounded_prompt(
                        "Rewrite the LinkedIn headline, About section, and experience bullets "
                        "for this target role. Keep every claim supported by evidence.", evidence
                    )], summary_type="detailed")
                st.markdown(result)
            except Exception as error:
                st.error(ErrorHandler.handle_error(
                    error, context="LinkedIn optimization"))
    with action_col2:
        if st.button("Generate cover letter", use_container_width=True):
            try:
                summarizer = PDFSummarizer()
                result = summarizer.summarize(
                    [grounded_prompt(
                        "Write a tailored cover letter using only the resume evidence and job "
                        "description. Mark unsupported details as [NEEDS CONFIRMATION].",
                        [f"Resume: {resume_text}",
                            f"Job description: {jd_text}"],
                    )], summary_type="detailed")
                st.markdown(result)
            except Exception as error:
                st.error(ErrorHandler.handle_error(
                    error, context="Cover letter generation"))
