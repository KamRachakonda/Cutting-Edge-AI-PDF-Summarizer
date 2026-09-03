# AI Career Intelligence

This extension turns the PDF summarizer into a career decision workspace while preserving the original PDF workflow.

## Capabilities
- Resume, JD/PD and supporting-document ingestion
- PDF, DOCX, TXT, Markdown, CSV, JSON and LOG parsing
- Semantic retrieval with ChromaDB + Sentence Transformers
- Source-aware evidence with document/chunk provenance
- Deterministic seven-dimension resume/JD fit scoring
- Document / web / hybrid question routing
- Optional Tavily research using external queries only
- Grounded career advice and factuality guardrails
- Resume positioning, LinkedIn optimization, cover letters and interview packs
- Downloadable Markdown/text career reports
- Original PDF OCR, statistics, charts, quotes and executive PDF export

## Environment
Copy `.env.example` to `.env` and configure `GROQ_API_KEY`. `TAVILY_API_KEY` is optional.

## Tests
Run `python -m unittest discover -s tests -v`.

## Design rule
Uploaded candidate documents are authoritative for candidate facts. External research provides contextual evidence. The deterministic score is calculated in Python; the LLM explains and recommends rather than inventing the score.
