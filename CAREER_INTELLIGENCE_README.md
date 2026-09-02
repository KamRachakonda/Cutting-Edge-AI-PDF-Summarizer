# AI Career Intelligence Platform

This branch upgrades the PDF summarizer into an evidence-grounded career intelligence application.

## Added capabilities

- Multi-file PDF, DOCX, TXT, Markdown, CSV, JSON and LOG ingestion
- Automatic Resume / Position Description classification
- Semantic RAG with ChromaDB + Sentence Transformers
- Structured resume and role analysis
- Weighted Resume-to-PD fit scoring
- Strength and gap analysis with evidence
- Career Advisor grounded in retrieved document evidence
- Optional external web research through Tavily
- Resume and executive positioning generation
- Interview preparation / STAR story generation
- Document summarisation

## Configuration

Create `.env` from `.env.example` and set `GROQ_API_KEY`.
`TAVILY_API_KEY` is optional; without it the application remains document-grounded and does not perform external research.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first RAG startup downloads the configured Sentence Transformer embedding model.

## Design principle

Candidate facts come only from uploaded documents. External research is treated as contextual evidence. If a capability is not supported by the candidate documents, the system should report it as a gap rather than invent experience.
